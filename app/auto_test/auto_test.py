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
from .config import auto_test_config
import traceback

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

# print("Using OpenAI API endpoint:", api_base)
# print("Using OpenAI API key:", api_key)
openai.api_type = "azure"
openai.api_base = api_base
openai.api_key = api_key
openai.api_version = "2023-05-15"
#%%

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
                print(role_color + role + ":\n" + message + Style.RESET_ALL)
            
            for message in self.messages:
                print_message(message["content"], message["role"])
        if auto_test_config.VERBOSE:
            return print(response)

        return response['choices'][0]['message']['content'] # type: ignore

# %%
PROMPTS = {
    "summarize": "You are an assistant, that takes code and tries to figure out what is the intent of the code. Be consie and try to answer with a single sentence.",
    
    "test": """You are an assistant that responds only with code. You given the code. Write a test for the code. Try to make tests based on properties and not specific values. (But you can use specific values if you want)
        It is very important that test function is called "test()" !!! Feel free to import any libraries if needed.
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
    It is very important that test function is called "test()" !!! Feel free to import any libraries if needed.
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
    It is very important that test function is called "test()" !!! Feel free to import any libraries if needed.
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
It is very important that test function is called "test()" !!! Feel free to import any libraries if needed.
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
    
        "try fixing": """You are an assistant that responds only with code. You given code that does not work. Try to fix the code.
        You, have access to code summary of the test, and the stack trace of the error.
        Write ONLY fixed code that was given in section Code . then finish with ```
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
        nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    return model


Test:
def test():
    model = create_conv_model()
    assert model is not None
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 32, 16, 16)

Stack trace:
...
    455 if self.padding_mode != 'zeros':
    456     return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
    457                     weight, bias, self.stride,
    458                     _pair(0), self.dilation, self.groups)
--> 459 return F.conv2d(input, weight, bias, self.stride,
    460                 self.padding, self.dilation, self.groups)

RuntimeError: Given groups=1, weight of size [32, 24, 3, 3], expected input[1, 16, 16, 16] to have 24 channels, but got 16 channels instead
    
Fixed code:
def create_conv_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # changed 24 to 16
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    return model

END
    """,
    
    "try fixing with intent": """You are an assistant that responds only with code. You given code that does not work. Try to fix the code.
        You, have access to code, summary and intent of the test, and the stack trace of the error.
        Write fixed code, no explanation needed, as code will be tested automatically.
        Write ONLY fixed code that was given in section Code . then finish with ```
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
        nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
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

Stack trace:
...
    455 if self.padding_mode != 'zeros':
    456     return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
    457                     weight, bias, self.stride,
    458                     _pair(0), self.dilation, self.groups)
--> 459 return F.conv2d(input, weight, bias, self.stride,
    460                 self.padding, self.dilation, self.groups)

RuntimeError: Given groups=1, weight of size [32, 24, 3, 3], expected input[1, 16, 16, 16] to have 24 channels, but got 16 channels instead
    
Fixed code:
def create_conv_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # changed 24 to 16
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    return model
    
END
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
    
def try_fixing_code(summary, code: str,test: str, stacktrace: str,intent: Optional[str]=None):
    
    if intent != None:
        conversation = Conversation(PROMPTS["try fixing with intent"])
        message = f"\n\nSummary:\n\n{summary}\n\nCode:\n\n{code}\n\nIntent:\n\n{intent}\n\nTest:\n\n{test}\n\nStack trace:\n\n{stacktrace}"
    else:
        conversation = Conversation(PROMPTS["try fixing"])
        message = f"\n\nSummary:\n\n{summary}\n\nCode:\n\n{code}\n\nTest:\n\n{test}\n\nStack trace:\n\n{stacktrace}"
    return conversation.user_message(message)

#%%
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function timed out")

def run_with_timeout(test,globalsV, timeout=5):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = exec(test,globalsV)
    except TimeoutException as e:
        result = None
    finally:
        signal.alarm(0)
    return result


def run_test(test: str,globalsV):

    try:
        # Define the test dynamically
        # run with timeout
        # use exec(test)
        # print("Running test \n")
        # print(test)
        run_with_timeout(test,globalsV, timeout=60)

    except Exception as e:
        # Return the error message
        # print(type(e))
        if type(e) == TimeoutException:
            return f"{Fore.RED}Test timed out{Style.RESET_ALL}"
        stack_trace = traceback.format_exc()
        #print(stack_trace)
        if "AssertionError" not in stack_trace:
            print(f"{Fore.RED}Test failed with error: {e}{Style.RESET_ALL}")
            return stack_trace
        # find last line that looks like this and extract xyz File "<string>", line xyz , ...
        processed_trace = stack_trace.split("\n")
        # filter lines that start with File "<string>", line xyz , ...
        print("Stack trace ",processed_trace)
        processed_trace = [line for line in processed_trace if line.startswith("  File \"<string>\", line ")]
        '  File "<string>", line 28, in test'
        print("Filtered ",processed_trace)
        # get last line
        processed_trace = processed_trace[-1]
        # remove File "<string>", line
        processed_trace = processed_trace.replace("  File \"<string>\", line ", "")
        # drop everything after ,
        processed_trace = processed_trace.split(",")[0]
        # parse to int
        line_with_error = int(processed_trace)

        # find line in test
        test = test.split("\n")
        test = test[line_with_error-1]
        
        # print in red assert that failed
        error = f"{Fore.RED}Assertion failed: {test}{Style.RESET_ALL}\n\n{line_with_error}"
        print(error)

        return stack_trace
    
    # green test passed
    return f"{Fore.GREEN}Test passed{Style.RESET_ALL}"
    

def clean_test(test: str):
    if "Test:" in test:
        test = test.split("Test:")[-1]


    # remove Fixed code:
    if "Fixed code:" in test:
        test = test.split("Fixed code:")[-1]


    if "```" in test:
        lines = test.split("\n")
        start = next(i for i, line in enumerate(lines) if line.startswith("```"))
        end = next(i for i, line in enumerate(lines[start+1:], start+1) if line.startswith("```"))
        test = "\n".join(lines[start+1:end])
    return test

def tester(intent: Optional[str] = None,globalsV={} ):
    def decorator(func):
        if not auto_test_config.TEST:
            return func
        
        code = inspect.getsource(func)

        # remove decorator find @tester and remove line with it
        line = next(i for i, line in enumerate(code.split("\n")) if "@tester" in line)
        code = "\n".join(code.split("\n")[line+1:])

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
        highlighted_code = highlight(code+ "\n\n"+test+ "\n\ntest()", PythonLexer(), TerminalFormatter())
        print(highlighted_code)

        #print(code+ "\n\n"+test+ "\n\ntest()")
        test_result = run_test(code+ "\n\n"+test+ "\n\ntest()",globalsV)
        # Blue summary, white text, 
        print(f"{Fore.BLUE}Test result:{Style.RESET_ALL}")
        print(test_result)

        if f"{Fore.GREEN}Test passed{Style.RESET_ALL}" in test_result:
            return func
        
        fixed_code = try_fixing_code(summary, code, test, test_result,intent)

        fixed_code = clean_test(fixed_code)

        print(f"{Fore.BLUE}Fixed code:{Style.RESET_ALL}")
        highlighted_code = highlight(fixed_code, PythonLexer(), TerminalFormatter())
        print(highlighted_code)

        test_result = run_test(fixed_code+ "\n\n"+test+ "\n\ntest()",globalsV)
        print(f"{Fore.BLUE}Test result:{Style.RESET_ALL}")
        print(test_result)

        if f"{Fore.GREEN}Test passed{Style.RESET_ALL}" in test_result:
            print(f"{Fore.GREEN}Try using code above{Style.RESET_ALL}")
            return func
        else:
            
            try:
                from IPython.display import Image, display
                ipython_available = True
            except ImportError:
                ipython_available = False

            if ipython_available:
                # print in yellow
                print(f"{Fore.YELLOW}We are sorry we couldn't fix your code, but here is a cat picture :){Style.RESET_ALL}")
                url = "https://cataas.com/cat"
                display(Image(url=url))
            else:
                print("We are sorry we couldn't fix your code, nor show you a cat picture :(")

        return func
    return decorator


# from auto_test import test
# from auto_test import auto_test_config
# from functools import lru_cache,cache
# auto_test_config.SUMMARIZE = True


# %%
# @tester()
# def boo(graph, source):
#     # Step 1: Prepare the distance and predecessor for each node
#     distance, predecessor = dict(), dict()
#     for node in graph:
#         distance[node], predecessor[node] = float('inf'), None
#     distance[source] = 0
    
#     # Step 2: Relax the edges
#     for _ in range(len(graph)):
#         for node in graph:
#             for neighbour in graph[node+1]:
#                 # If the distance between the node and the neighbour is lower than the current, store it
#                 if distance[neighbours] > distance[node] + graph[node][neighbour]:
#                     distance[neighbour], predecessor[neighbour] = distance[node] + graph[node][neighbour], node

#     return distance, predecessor

# %%
