import sys
import logging

# Function to extract detailed error message
def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()  # Get the traceback
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where the error occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  # Get the error message details
    )
    return error_message

# Custom Exception Class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message_detail(error_message, error_detail=error_detail)  # Call function to get error details
        super().__init__(self.error_message)  # Initialize the Exception class with the formatted error message
    
    def __str__(self):
        return self.error_message  # Return the error message when printed

