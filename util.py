'''
Util class
'''

import os
import pandas as pd

os.system('cls' if os.name == 'nt' else 'clear')

class Util():
    def __init__(self):
        self.features = {'JourneyCode':0, 'HolidayType':1, 'Price':2, 'NumberOfPersons':3, 'Region':4,
                'Transportation':5, 'Duration':6, 'Season':7, 'Accommodation':8, 'Hotel':9}
    
    def build_df(self, input_file):
        with open(input_file, 'r+') as f:
            lines = f.readlines()
        travel_cb_dict={}
        for i in range(len(lines)):
                if self.is_case_object(lines[i]):
                    case_key=self.get_case_key(lines[i])
                    feature_list=[None]*len(self.features)
                    i+=1
                    while self.is_feature(lines[i]):
                        feature_index=self.features[self.get_feature_key(lines[i])]
                        feature_value=self.get_feature_value(lines[i], self.get_feature_key(lines[i]))
                        feature_list[feature_index]=feature_value
                        i+=1
                    travel_cb_dict[case_key]=feature_list
        df = pd.DataFrame(travel_cb_dict.values(), columns = self.features.keys())
        df.to_csv('data/travel_cb_orig.csv', encoding='utf-8', index=False)

        self.holiday_types = {k: v+1 for v, k in enumerate(sorted(df['HolidayType'].unique()))}
        self.regions={k: v+1 for v, k in enumerate(sorted(df['Region'].unique()))}
        self.transportation_modes={k: v+1 for v, k in enumerate(sorted(df['Transportation'].unique()))}
        self.seasons={k: v+1 for v, k in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
                'December'])}
        self.accomodations={k: v+1 for v, k in enumerate(['HolidayFlat', 'OneStar', 'TwoStars', 'ThreeStars', 'FourStars', 'FiveStars'])}
        self.hotels={k: v+1 for v, k in enumerate(df['Hotel'].unique())}

        travel_cb_dict_new={}
        for k,v in travel_cb_dict.items():
            travel_cb_dict_new[k]=self.standardize_feature_list(v)
        df = pd.DataFrame (travel_cb_dict_new.values(), columns = self.features.keys())
        return df.astype(int)

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
    
    def standardize_feature_list(self, featurelist):
        featurelist[1]=self.holiday_types[featurelist[1]]
        featurelist[4]=self.regions[featurelist[4]]
        featurelist[5]=self.transportation_modes[featurelist[5]]
        featurelist[7]=self.seasons[featurelist[7]]
        featurelist[8]=self.accomodations[featurelist[8]]
        featurelist[9]=self.hotels[featurelist[9]]
        return featurelist