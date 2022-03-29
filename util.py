'''
Util class
'''

import os
os.system('cls' if os.name == 'nt' else 'clear')

class Util():
    def __init__(self, features):
        self.features=features

    def is_case_object(self, input_line)->bool:
        '''checks if input line is a case object'''
        return bool(len(input_line)-len(input_line.lstrip('\t'))==2
                    and input_line.strip().split()[0]=='case')

    def is_feature(self, input_line)->bool:
        '''checks if input line is a feature element'''
        return bool(len(input_line)-len(input_line.lstrip('\t'))==3
                    and input_line.strip().split()[0].strip(':') in self.features)
    
    def get_case_key(self, input_line):
        return input_line.strip().split()[1]

    def get_feature_key(self, input_line):
        return input_line.strip().split()[0].strip(':')

    def get_feature_value(self, input_line, featurekey):
        return input_line.strip().removeprefix(featurekey).strip(':,. "')

if __name__=='__main__':
    input_file='data/travel_cb.txt'
    features = {'JourneyCode':0, 'HolidayType':1, 'Price':2, 'NumberOfPersons':3, 'Region':4,
                'Transportation':5, 'Duration':6, 'Season':7, 'Accommodation':8, 'Hotel':9}
    with open(input_file, 'r+') as f:
        lines = f.readlines()

    util=Util(features)

    travel_cb_dict={}

    for i in range(len(lines)):
        if util.is_case_object(lines[i]):
            case_key=util.get_case_key(lines[i])
            feature_list=[None]*len(features)
            i+=1
            while util.is_feature(lines[i]):
                feature_index=features[util.get_feature_key(lines[i])]
                feature_value=util.get_feature_value(lines[i], util.get_feature_key(lines[i]))
                feature_list[feature_index]=feature_value
                i+=1
            travel_cb_dict[case_key]=feature_list

    print(travel_cb_dict['Journey1470'])
    print(len(travel_cb_dict))