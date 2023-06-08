class AutoTestConfig:
    SUMMARIZE = True # if True, will use summarization to create prompt
    TEST = True # if False, will not run tests, nor query API
    DEBUG = False # if True, will print out prompts and responses
    VERBOSE = False # if True, will print out raw API responses
    SAVE_TESTS = False # if True, will save tests to file for inspection


    colors = {
        "user": "\033[94m", # blue
        "system": "\033[92m", # green
        "assistant": "\033[91m", # red
        "debug": "\033[93m", # yellow
    }

    def __init__(self, test_mode: bool = True, debug_mode: bool = False, verbose: bool = False, summarize: bool = False, save_tests: bool = True):
        self.TEST_MODE = test_mode
        self.DEBUG_MODE = debug_mode
        self.VERBOSE = verbose
        self.SUMMARIZE = summarize
        self.SAVE_TESTS = save_tests

    def set_test_mode(self, test_mode: bool):
        "If true, tests are run. Every time function is defined, it is run with the test data."
        self.TEST_MODE = test_mode

    def set_debug_mode(self, debug_mode: bool):
        "If true, debug messages are printed."
        self.DEBUG_MODE = debug_mode
    
    def set_verbose(self, verbose: bool):
        "If true, full responses are printed."
        self.VERBOSE = verbose
    
    def set_summarize(self, summarize: bool):
        "If true, the assistant will summarize the code."
        self.SUMMARIZE = summarize

    def set_save_tests(self, save_tests: bool):
        "If true, the assistant will save the tests."
        self.SAVE_TESTS = save_tests

    def __repr__(self) -> str:
        return f"AutoTestConfig(test_mode={self.TEST_MODE}, debug_mode={self.DEBUG_MODE}, verbose={self.VERBOSE}, summarize={self.SUMMARIZE}, save_tests={self.SAVE_TESTS})"

auto_test_config = AutoTestConfig()

