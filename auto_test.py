#%%#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
from typing import List, Dict,Optional
import inspect
import json
import os
from colorama import Fore, Back, Style
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
CONFIG_FILE = "openai_config.json"

def get_openai_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        return {}

def set_openai_config(api_base, api_key):
    config = {"api_base": api_base, "api_key": api_key}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

config = get_openai_config()
api_base = ""
api_key = ""
if "api_base" in config and "api_key" in config:
    api_base = config["api_base"]
    api_key = config["api_key"]
if "AZURE_OPENAI_ENDPOINT" in os.environ and "AZURE_OPENAI_API_KEY" in os.environ:
    api_base = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]

# if none
if api_base == "":
    api_base = input("Enter OpenAI API endpoint: ")
    
if api_key == "":
    api_key = input("Enter OpenAI API key: ")

set_openai_config(api_base, api_key)

assert api_base !=  ""
assert api_key !=  ""

print("Using OpenAI API endpoint:", api_base)
print("Using OpenAI API key:", api_key)
openai.api_type = "azure"
openai.api_base = api_base
openai.api_key = api_key
openai.api_version = "2023-05-15"
#%%
class AutoTestConfig:
    SUMMARIZE = False
    TEST = True
    DEBUG = False
    VERBOSE = False
    SAVE_TESTS = True
    

    colors = {
        "user": "\033[94m", # blue
        "system": "\033[92m", # green
        "assistant": "\033[91m", # red
        "debug": "\033[93m", # yellow
    }

    def __init__(self, test_mode: bool = True, debug_mode: bool = False, verbose: bool = False):
        self.TEST_MODE = test_mode
        self.DEBUG_MODE = debug_mode
        self.VERBOSE = verbose

    def set_test_mode(self, test_mode: bool):
        "If true, tests are run. Every time function is defined, it is run with the test data."
        self.TEST_MODE = test_mode

    def set_debug_mode(self, debug_mode: bool):
        "If true, debug messages are printed."
        self.DEBUG_MODE = debug_mode
    
    def set_verbose(self, verbose: bool):
        "If true, full responses are printed."
        self.VERBOSE = verbose

auto_test_config = AutoTestConfig()
#%%
class Conversation:
    # list of dicts
    messages: List[Dict] = [] 

    def __init__(self,system_message: str):
        self.messages = [{"role": "system", "content": system_message}]

    def user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            engine="cim", # engine = "deployment_name".
            messages=self.messages
        )
        self.messages.append(response['choices'][0]['message'].to_dict()) # type: ignore
        if auto_test_config.DEBUG:
            # use colorama to color the output
            
            def print_message(message: str, role: str):
                role_color = auto_test_config.colors[role]
                print(role_color + role + ": " + message + Style.RESET_ALL)
            
            for message in self.messages:
                print_message(message["content"], message["role"])
        if auto_test_config.VERBOSE:
            return print(response)

        return response['choices'][0]['message']['content'] # type: ignore

# %%
PROMPTS = {
    "summarize": "You are an assistant, that takes code and tries to figure out what is the intent of the code. Be consie and try to answer with a single sentence.",
    
    "test": """You are an assistant that responds only with code. You given the code. Write a test for the code. Try to make tests based on properties and not specific values. (But you can use specific values if you want)
    Example:
User: 
Code:
def sqrt(x):
    return x ** 0.5
Test:
def test():
    for i in range(100):
        assert abs(sqrt(i)**2 - i) < 1e-6 
    
    """,

    "test with intent": """You are an assistant that responds only with code. You write tests. Code and user intention of user of what needs to be tested. Write a test for the code.
Example:
User: 
Code:
def create_conv_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    return model
Intent:
    Check shape of the output of the model to be (, 32, 16, 16)
Test:
def test():
    model = create_conv_model()
    assert model is not None
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 32, 16, 16)
        
    """,
    
    "test with summary": """You are an assistant that responds only with code. You given summary of the code and implementation. Write a test for the code. Try to make tests based on properties and not specific values. (But you can use specific values if you want)
    Example:
User: 
Summary:
    This function computes root of a number
Code:
def sqrt(x):
    return x ** 0.5
Test:
def test():
    for i in range(100):
        assert abs(sqrt(i)**2 - i) < 1e-6 
    
    """,

    "test with summary and intent": """You are an assistant that responds only with code. You write tests. Given summary of the code, implementation and user intention of user of what needs to be tested. Write a test for the code.
Example:
User: 

Summary:
    This function creates a convolutional neural network model
Code:
def create_conv_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    return model
Intent:
    Check shape of the output of the model to be (, 32, 16, 16)
Test:
def test():
    model = create_conv_model()
    assert model is not None
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 32, 16, 16)
        
    """,
    
}

def summarize_code(code: str):
    conversation = Conversation(PROMPTS["summarize"])
    return conversation.user_message("\n\nCode:\n\n"+ code)

# prompt options
"test"
"test with intent"
"test with summary"
"test with summary and intent"
def test_generic(code: str,summary: Optional[str] = None,intent: Optional[str] = None):
    # select right prompt
    if summary == None and intent == None:
        prompt = PROMPTS["test"]
    elif summary == None and intent != None:
        prompt = PROMPTS["test with intent"]
    elif summary != None and intent == None:
        prompt = PROMPTS["test with summary"]
    elif summary != None and intent != None:
        prompt = PROMPTS["test with summary and intent"]
    else:
        raise ValueError("Wow, you broke the matrix")

    conversation = Conversation(prompt)
    
    # construct message Summary then Code then Intent
    message = ""
    if summary != None:
        message += "\n\nSummary:\n\n"+ summary
    message += "\n\nCode:\n\n"+ code
    if intent != None:
        message += "\n\nIntent:\n\n" + intent
    
    return conversation.user_message(message)
    

#%%
def run_test(test: str):

    try:
        # Define the test dynamically
        exec(test)

    except Exception as e:
        # Return the error message
        return str(e)
    
    # green test passed
    return f"{Fore.GREEN}Test passed{Style.RESET_ALL}"
    

def clean_test(test: str):
    if test.startswith("Test:\n"):
        test = test[6:]

    if "```" in test:
        lines = test.split("\n")
        start = next(i for i, line in enumerate(lines) if line.startswith("```"))
        end = next(i for i, line in enumerate(lines[start+1:], start+1) if line.startswith("```"))
        test = "\n".join(lines[start+1:end])
    return test

def auto_test(intent: Optional[str] = None ):
    def decorator(func):
        if not auto_test_config.TEST:
            return func
        code = inspect.getsource(func)
        code = code.replace("@auto_test()\n", "")

        if auto_test_config.SUMMARIZE:
            summary = summarize_code(code)
            print(f"{Fore.BLUE}Summary:{Style.RESET_ALL}")
            print(summary)
        else:
            summary = None
        
        test = test_generic(code, summary, intent)

        test = clean_test(test)

        if auto_test_config.SAVE_TESTS: 
            # save test for easy inspection
            with open("test_tmp.py", "w") as f:
                f.write(test)

        print(f"{Fore.BLUE}Test:{Style.RESET_ALL}")
        highlighted_code = highlight(test, PythonLexer(), TerminalFormatter())
        print(highlighted_code)

        test_result = run_test(test)
        # Blue summary, white text, 
        print(f"{Fore.BLUE}Test result:{Style.RESET_ALL}")
        print(test_result)

        return func
    return decorator


# %%
from functools import lru_cache,cache

@auto_test()
@cache
def fib(n):
    if n < 2:
        return n
    return fib(n - 2) + fib(n - 1)
# %%
fib(5)
