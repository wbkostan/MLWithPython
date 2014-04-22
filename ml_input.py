from os.path import exists

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg

class MLInputFile:
    def __init__(self, filename):
        self.number_of_records = None
        self.headers = []
        self.data = {}
        if exists(filename):
            self.filename = filename
            self.__extract_data__()
        else:
            raise FileNotFoundError

    def __extract_data__(self):
        with open(self.filename, 'r') as file:
            header = file.readline()
            self.__process_header__(header)
            for i in range(0,self.number_of_records):
                line = file.readline()
                self.__process_line__(line)

    def __process_header__(self, header):
        #tab delimited
        components = header.split(chr(9))
        if not len(components) == 2:
            raise InputError(header, "Input header not recognized")
        split_N = components[0].split('=')
        self.number_of_records = int(split_N[1])
        split_H = components[1].strip().split('=')
        if not (split_H[1][-1] == ']' and split_H[1][0] == '['):
            raise InputError(header, "Header lables must be input as an array")
        else:
            #strip brackets, remove whitespace, then split on ',' into an array
            self.headers = split_H[1][1:-1].replace(" ", "").split(',')

    def __process_line__(self, line):
        #tab delimited
        components = line.split(chr(9))
        for header, component in zip(self.headers, components):
            if not header in self.data:
                self.data[header] = []
            self.data[header].append(component)

