from os.path import exists
from custom_exceptions import InputError
from random import shuffle
from math import floor

class MLInputFile:
    def __init__(self, filename):
        self.number_of_records = None
        self.headers = []
        self.raw_data = {}
        self.vocabulary = {}
        if exists(filename):
            self.filename = filename
            self.__extract_data__()
            self.__randomize_data__()
        else:
            raise FileNotFoundError

    def __extract_data__(self):
        with open(self.filename, 'r') as file:
            header = file.readline()
            self.__process_header__(header)
            for i in range(0,self.number_of_records):
                line = file.readline().strip()
                self.__process_line__(line)

    def __randomize_data__(self):
        keys = list(self.raw_data.keys())
        assert(keys != [])
        data_indices = list(range(self.number_of_records))
        shuffle(data_indices)
        new_data = {}
        for key in keys:
            new_data[key] = []
        for i in data_indices:
            for key in keys:
                new_data[key].append(self.raw_data[key][i])
        self.raw_data = new_data

    def split_data_set(self, percent_split):
        keys = list(self.raw_data.keys())
        assert(keys != [])
        split_index = floor(percent_split*self.number_of_records)
        train = {}
        test = {}
        for key in keys:
            train[key] = self.raw_data[key][:split_index]
            test[key] = self.raw_data[key][split_index:]
        return train,test


    def __process_header__(self, header):
        #tab delimited
        components = header.split(chr(9))
        if not len(components) == 2:
            raise InputError(header, "Input header not recognized")
        split_N = components[0].split('=')
        self.number_of_records = int(split_N[1])
        split_H = components[1].strip().split('=')
        if not (split_H[1][-1] == ']' and split_H[1][0] == '['):
            raise InputError(header, "Header labels must be input as an array")
        else:
            #strip brackets, remove whitespace, then split on ',' into an array
            self.headers = split_H[1][1:-1].replace(" ", "").split(',')

    def __process_line__(self, line):
        #tab delimited
        components = line.split(chr(9))
        for header, component in zip(self.headers, components):
            if not header in self.raw_data:
                self.raw_data[header] = []
            self.raw_data[header].append(component)

    def __vocabularize__(self):
        for index, cat in enumerate(set(self.raw_data['categoryLabel'])):
            self.vocabulary[cat] = index
            self.vocabulary[index] = cat