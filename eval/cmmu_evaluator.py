import json
import sys
import ast
import re
import openai
import pickle
import difflib
from collections import defaultdict
import os.path as osp
from eval.prompts import EVALUATION_SYSTEM_PROMPT, EVALUATION_USER_TEMPLATE
from eval.pre_process import process_multiple_choice, normalize_string


class ExamEvaluator:
    def __init__(
        self,
        output_dir,
        result_file_name,
        filter_types=None,
        shift_check=False,
        gpt_model_name=None,
        is_clean=True,
    ) -> None:
        if gpt_model_name is not None:
            from .chat_llm import ChatLLM

            self.llm = ChatLLM(
                chat_name="exam", use_cache=True, model_name=gpt_model_name
            )
            self.evaluate_fill_blank = self.evaluate_fill_blank_llm
        else:
            print("Evaluate by rules")
            self.evaluate_fill_blank = self.evaluate_fill_blank_by_rule
        self.output_dir = output_dir
        self.result_file_name = result_file_name
        if isinstance(filter_types, list):
            self.filter_types = set(filter_types)
        else:
            self.filter_types = None
        self.is_clean = is_clean
        self.shift_check = shift_check
        self.evaluated_ids_count = defaultdict(list)  # only used for circular_eval
        self.grade = ["primary", "middle", "high"]
        self.subject = [
            "math",
            "politics",
            "history",
            "biology",
            "chemistry",
            "geography",
            "physics",
        ]
        self.difficulty = ["normal", "hard"]
        self.score_dict = self.init_score_dict()
        self.results = []

    def init_score_dict(self):
        score_dict = {}
        for g in self.grade:
            for s in self.subject:
                for d in self.difficulty:
                    score_dict[f"{g}-{s}-{d}"] = [0, 0]
        return score_dict

    def evaluate_fill_blank_llm(self, gt, answer, key):
        prompt = EVALUATION_USER_TEMPLATE.format(
            gt["question_info"], gt["answer"], answer["answer"]
        )
        messages = [
            {
                "role": "system",
                "content": EVALUATION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        print(f"gt: {gt['answer']}\nans: {answer['answer']}")
        self.score_dict[key][1] += 1
        try:
            ans = self.llm.chat(messages)
        except openai.BadRequestError as e:
            print(e)
            return False, answer["answer"]
        try:
            ans_parsed = ast.literal_eval(ans)
            self.score_dict[key][0] += ans_parsed["correct"]
            return ans_parsed["correct"], answer["answer"]
        except Exception as e:
            pattern = re.compile(r'"correct"\s*:\s*1')
            match = re.search(pattern, ans)
            if match:
                return True, answer["answer"]
            return False, answer["answer"]

    def maybe_clean_answer(self, answer):
        if not self.is_clean:
            return answer
        if len(answer) == 1:
            return answer
        answer = process_multiple_choice(answer)
        return answer

    def evaluate_mulitple_choice(self, gt, answer, key):
        cleaned_answer = self.maybe_clean_answer(answer["answer"])
        is_right = gt["answer"] == cleaned_answer[:1]
        if not self.shift_check:
            self.score_dict[key][0] += is_right
            self.score_dict[key][1] += 1
            return is_right
        question_id = str(answer["question_id"])
        question_id_base = question_id.split("-")[0]

        self.evaluated_ids_count[question_id_base].append(is_right)

        if len(self.evaluated_ids_count[question_id_base]) == len(gt["options"]):
            self.score_dict[key][0] += len(gt["options"]) == sum(
                self.evaluated_ids_count[question_id_base]
            )
            self.score_dict[key][1] += 1
        return is_right, cleaned_answer

    def evaluate_multiple_response(self, gt, answer, key):
        cleaned_answer = self.maybe_clean_answer(answer["answer"])
        is_right = gt["answer"] == "".join(sorted(cleaned_answer))
        self.score_dict[key][0] += is_right
        self.score_dict[key][1] += 1
        return is_right, cleaned_answer

    def show(self, content):
        print(content)
        self.results.append(content)

    def evaluate_fill_blank_by_rule(self, gt, answer, key, simality_threshold=0.7):
        splited_answer = answer["answer"].split("\n")
        cleaned_answers = []
        self.score_dict[key][1] += 1
        for raw_answer in splited_answer:
            cleaned_answers.append(normalize_string(raw_answer))
        gt = normalize_string(gt["answer"])
        for cleaned_answer in cleaned_answers:
            simality = difflib.SequenceMatcher(None, cleaned_answer, gt).ratio()
            if simality > simality_threshold:
                self.score_dict[key][0] += 1
                return True, cleaned_answer
        return False, "\n".join(cleaned_answers)

    def cal_accuracy(self, annotation, answers, target_type=None):
        self.score_dict = self.init_score_dict()
        if target_type is not None:
            self.show(f"\nEvaluate {target_type}")
        else:
            self.show(f"\nEvaluate all types of questions")
        for answer in answers:
            question_id = str(answer["question_id"])
            if "type" in answer and answer["type"] not in self.filter_types:
                continue
            if question_id not in annotation:
                continue
            gt = annotation[question_id]
            key = f'{gt["grade_band"]}-{gt["subject"]}-{gt["difficulty"]}'

            if target_type is not None and gt["type"] != target_type:
                continue
            if gt["type"] == "fill-in-the-blank":
                is_right, cleaned_answer = self.evaluate_fill_blank(gt, answer, key)
                cleaned_answer = cleaned_answer
            elif gt["type"] == "multiple-choice":
                is_right, cleaned_answer = self.evaluate_mulitple_choice(
                    gt, answer, key
                )
            else:
                is_right, cleaned_answer = self.evaluate_multiple_response(
                    gt, answer, key
                )
            answer["correct"] = is_right
            answer["label"] = gt["answer"]
            answer["answer_raw"] = answer["answer"]
            answer["answer"] = cleaned_answer

        # show results
        subject_right = defaultdict(int)
        subject_total = defaultdict(int)
        grade_total = defaultdict(int)
        grade_right = defaultdict(int)
        difficulty_total = defaultdict(int)
        difficulty_right = defaultdict(int)
        eps = sys.float_info.epsilon
        for g in self.grade:
            for s in self.subject:
                ques_num = 0
                right_num = 0
                for d in self.difficulty:
                    key = f"{g}-{s}-{d}"
                    if self.score_dict[key][1] > 0:
                        self.show(
                            f"{key}: {self.score_dict[key][0] / self.score_dict[key][1] :.5f} ({self.score_dict[key][0]} / {self.score_dict[key][1]})"
                        )
                    ques_num += self.score_dict[key][1]
                    right_num += self.score_dict[key][0]
                    subject_right[s] += self.score_dict[key][0]
                    subject_total[s] += self.score_dict[key][1]
                    grade_right[g] += self.score_dict[key][0]
                    grade_total[g] += self.score_dict[key][1]
                    difficulty_right[d] += self.score_dict[key][0]
                    difficulty_total[d] += self.score_dict[key][1]
                if ques_num > 0:
                    self.show(
                        f"{g}-{s}-Total: {right_num / ques_num :.5f} ({right_num} / {ques_num})\n"
                    )

        self.show("===Grade Total===")
        for g in self.grade:
            self.show(f"{g}: {grade_right[g] / (grade_total[g] + eps):.5f}")
        self.show("===Subject Total===")
        for s in self.subject:
            self.show(f"{s}: {subject_right[s] / (subject_total[s] + eps):.5f}")
        self.show("===Diffucility Total==")
        total_right, total_questions = 0, 0
        for d in self.difficulty:
            total_right += difficulty_right[d]
            total_questions += difficulty_total[d]
            self.show(f"{d}: {difficulty_right[d] / (difficulty_total[d] + eps):.5f}")
        self.show(f"==={target_type} Accuracy Overall===")
        self.show(f"Overall: {total_right / (total_questions + eps):.5f}")
        return total_right, total_questions

    def dump_results(self, judged_answers):
        print("\n\n=======Final Results=======")
        data = "\n".join(self.results)
        print(data)
        with open(
            osp.join(osp.join(self.output_dir, self.result_file_name + ".summary")), "w"
        ) as fout:
            fout.write(data)
        with open(
            osp.join(self.output_dir, self.result_file_name + "_judged.json"), "w"
        ) as fout:
            json.dump(judged_answers, fout, ensure_ascii=False, indent=2)

    def dump_score_dict(self, ques_type):
        pickle.dump(
            self.score_dict,
            open(
                osp.join(
                    self.output_dir,
                    f"{self.result_file_name}_{ques_type}_score_dict.pkl",
                ),
                "wb",
            ),
        )

    def process(self, dataset, result_path):
        """
        Args:
            dataset (Dataset): dataset instance
            answers (list): list of answers
        """
        annotation = dataset.get_annotation()
        dataset_name = dataset.name
        if osp.isfile(result_path):
            result_file = result_path
        else:
            result_file = osp.join(result_path, dataset_name + ".json")
        answers = json.load(open(result_file))

        if self.filter_types is None:
            results = self.cal_accuracy(annotation, answers)
            self.dump_results(answers)
        else:
            results = {}
            for ques_type in self.filter_types:
                results[ques_type] = self.cal_accuracy(annotation, answers, ques_type)
                self.dump_score_dict(ques_type)
            self.dump_results(answers)
            return results
