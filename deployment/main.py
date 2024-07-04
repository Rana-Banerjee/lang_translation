"""Kserve inference script."""

import argparse
import re

from kserve import (
    InferOutput,
    InferRequest,
    InferResponse,
    Model,
    ModelServer,
    model_server,
)
from kserve.utils.utils import generate_uuid
# from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast
import torch
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer

MODEL_DIR = "./saved_model/"
FAIRSEQ_LANGUAGE_CODES = ['dyu_Latn', 'fr_Latn']
mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]
def get_non_printing_char_replacer(replace_by: str = " "):
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

# PREFIX = "translate Dyula to French: "  # Model's inference command
# MODEL_KWARGS = {
#     "do_sample": True,
#     "max_new_tokens": 40,
#     "top_k": 30,
#     "top_p": 0.95,
#     "temperature": 1.0,
# }

# CHARS_TO_REMOVE_REGEX = '[!"&\(\),-./:;=?+.\n\[\]]'
# def clean_text(text: str) -> str:
#     """
#     Clean input text by removing special characters and converting
#     to lower case.
#     """
#     text = re.sub(CHARS_TO_REMOVE_REGEX, " ", text.lower())
#     return text.strip()


class MyModel(Model):
    """Kserve inference implementation of model."""

    def __init__(self, name: str):
        """Initialise model."""
        super().__init__(name)
        self.name = name
        self.model = None
        self.tokenizer = None
        self.ready = False
        if torch.cuda.is_available():
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.load()

    def load(self):
        """Reconstitute model from disk."""
        # Load model and tokenizer
        self.tokenizer = NllbTokenizerFast.from_pretrained(MODEL_DIR, truncation=True,additional_special_tokens=FAIRSEQ_LANGUAGE_CODES)
        self.tokenizer.src_lang='dyu_Latn'
        self.tokenizer.tgt_lang='fra_Latn'
        self.model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR,torch_dtype=torch.float16).to(self.device_type).eval()
        self.ready = True

    def preprocess(self, payload: InferRequest, *args, **kwargs) -> str:
        """Preprocess inference request."""
        # Clean input sentence and add prefix
        raw_data = payload.inputs[0].data[0]
        # prepared_data = f"{PREFIX}{clean_text(raw_data)}"
        prepared_data = mpn.normalize(raw_data)
        prepared_data = replace_nonprint(prepared_data)
        prepared_data = unicodedata.normalize("NFKC", prepared_data)
        return prepared_data

    def predict(self, data: str, *args, **kwargs) -> InferResponse:
        """Pass inference request to model to make prediction."""
        # Model prediction preprocessed sentence
        inference_input = self.tokenizer(data, return_tensors="pt", truncation=True).to(self.device_type)
        output = self.model.generate(**inference_input, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids('fra_Latn'), max_length=140)
        translation = self.tokenizer.decode(output[0], skip_special_tokens=True)
        response_id = generate_uuid()
        infer_output = InferOutput(
            name="output-0", shape=[1], datatype="STR", data=[translation]
        )
        infer_response = InferResponse(
            model_name=self.name, infer_outputs=[infer_output], response_id=response_id
        )
        return infer_response


parser = argparse.ArgumentParser(parents=[model_server.parser])
parser.add_argument(
    "--model_name", default="model", help="The name that the model is served under."
)
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = MyModel(args.model_name)
    ModelServer().start([model])