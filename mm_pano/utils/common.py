import os
import re
import json
from dataclasses import dataclass, field
from typing import Optional, List


def extract_words_after_we_see_withFailv2(s):
    match = re.search('We .*?see: (.*)', s, re.IGNORECASE)
    if match:
        return match.group(1).replace('.', '').lower()
    print("No match found")
    return


def extract_words_after_we_see_withFailv3(s):
    match = re.search('We .*?see(.*)', s, re.IGNORECASE) or re.search('View .*?:(.*)', s, re.IGNORECASE)
    if match:
        return match.group(1)
    print("No match found")
    return


@dataclass
class Descriptor:
    generated_text_details: Optional[str] = None
    message: Optional[str] = None
    message_main_obj: Optional[str] = None
    message_topdown: Optional[str] = None
    question_for_llm_repeat: Optional[str] = None
    description_no_obj: Optional[str] = None
    major_obj_number: int = 2
    is_repeated: List[bool] = field(default_factory=list)

    init_prompt: Optional[str] = None
    init_image: Optional[str] = None

    @classmethod
    def from_json(cls, json_path: str):
        assert isinstance(json_path, str) and os.path.isfile(json_path)
        with open(json_path, "r") as f:
            _dict = json.load(f)
        print(_dict)
        return cls(**_dict)

    def save_json(self, json_path: str):
        assert isinstance(json_path, str)
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def __post_init__(self):
        assert self.init_prompt is not None or self.init_image is not None, \
            "When using Descriptor, either `init_prompt` or `init_image` has to be set. Got both None."

        if self.init_prompt is not None and self.init_image is not None:
            print(f"Both `init_prompt` ({self.init_prompt}) and `init_image` ({self.init_image}) "
                  " is given, using `init_image` and ignore `init_prompt`")
            self.init_prompt = None

        if self.init_image:
            assert os.path.isfile(self.init_image), f"The given `init_image` is not a valid file {self.init_image}"
