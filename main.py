import pandas as pd
import util
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from collections import OrderedDict

class ccbr():
    def __init__(self, input_file) -> None:
        self.util_obj=util.Util()
        self.orig_df=pd.read_csv('data/travel_cb_orig.csv', encoding='utf-8')
        self.df=self.util_obj.build_df(input_file)
        self.df.to_csv('data/travel_cb.csv', encoding='utf-8')
        
        self.predictors=[
            'HolidayType','NumberOfPersons','Region', 'Transportation', 'Duration', 'Season', 
            'Accommodation','Hotel'
        ]
        xtrain, xtest, ytrain, ytest = train_test_split(self.df[self.predictors],self.df['Price'], 
                                                  random_state=42, 
                                                  test_size=0.20, shuffle=True)
        self.reg = LinearRegression().fit(xtrain, ytrain)
        y_pred_train = self.reg.predict(xtrain)
        y_pred_test = self.reg.predict(xtest)
        coef= np.abs(self.reg.coef_)
        score = coef/sum(coef)
        self.score_dict = {}
        score = list(coef/sum(coef))
        for i in range(len(self.predictors)):
            self.score_dict[self.predictors[i]] = score[i] 
        self.score_dict = {k: v for k, v in sorted(self.score_dict.items(), key=lambda item: item[1],reverse=True)}
        self.running_feature_ranklist=list(self.score_dict.keys())

        self.nominal_features=['Region', 'Hotel']
        self.f_q_map={
            'HolidayType':'What holiday type are you looking for? ',
            'NumberOfPersons':'How many people are there in the trip? ',
            'Region':'Which region are you interested in travelling? ',
            'Transportation':'What mode of transportation are you looking for? ',
            'Duration':'What is the duration of the trip? ',
            'Season':'In which Season are planning the trip? ',
            'Accommodation':'What type of accomodation you want?',
            'Hotel':'Do you have any preference of Hotel?'
        }
    
        self.f_q_map_detailed={
            'HolidayType':('What holiday type are you looking for?\n'
                'Choose 0 for Active\nChoose 1 for Bathing\nChoose 2 for City\n'
                'Choose 3 for Education\nChoose 4 for Language\nChoose 5 for Recreation\n'
                'Choose 6 for Skiing\nChoose 7 for Wandering\n'),
            'NumberOfPersons':'How many people are there in the trip? ',
            'Region':'Which region are you interested in travelling? ',
            'Transportation':('What mode of transportation are you looking for?\n'
                'Choose 0 for Car\nChoose 1 for Coach\nChoose 2 for Plane\n'
                'Choose 3 for Train\n'),
            'Duration':'What is the duration of the trip?\n',
            'Season':('In which Season are planning the trip?\n'
                'Choose 1 for January, 2 for February ... 12 for December\n'),
            'Accommodation':('What type of accomodation you want?\n'
                'Choose 0 for HolidayFlat\nChoose 1 for OneStar\nChoose 2 for TwoStars\n'
                'Choose 3 for ThreeStars\nChoose 4 for FourStars\nChoose 5 for FiveStars\n'),
            'Hotel':'Do you have any preference of Hotel?\n'
        }
        self.user_pref={'NumberOfPersons':3, 'Season':6}

    def start(self):
        while self.running_feature_ranklist:
            selected_feature=self.get_q_preference()
            selected_feature_val=self.get_feature_val(selected_feature)
            self.user_pref[selected_feature]=selected_feature_val
            self.user_pref=self.get_standard_feature_dict(self.user_pref)
            best_case_ids=self.get_similar_cases(self.user_pref)
            is_final=self.check_case_with_user(best_case_ids)
            if is_final:
                self.print_final_price()
                break
            else:
                pass
        if not is_final:
            self.print_final_price(self.adapt_case(self.user_pref))
        
    def get_q_preference(self):
        for i in range(3):
            try:
                ques=self.f_q_map.get(self.running_feature_ranklist[i])
            except IndexError:
                break
            print(f'{i+1}: {ques} (Score: {self.score_dict.get(self.running_feature_ranklist[i])})')
        selected_feature_id=int(input('Select one of the above Qs (1, 2 or 3...)'))
        selected_feature_name=self.running_feature_ranklist[selected_feature_id-1]
        self.running_feature_ranklist.pop(selected_feature_id-1)
        # self.update_feature_ranking(selected_feature_name)
        return selected_feature_name

    def get_feature_val(self, selected_feature):
        if selected_feature not in self.nominal_features:
            return int(input(self.f_q_map_detailed.get(selected_feature)))
        else:
            return input(self.f_q_map.get(selected_feature))
    
    def get_similar_cases(self, user_pref, k=3):
        # retrieve indices of 3 similar cases
        ## just for testing
        return np.random.randint(1,1471,3)

    def check_case_with_user(self, best_case_ids):
        # print the cases to the user
        for id in best_case_ids:
            print(f'{self.orig_df.iloc[id]}')
        is_final=input('Do you want to select a Journey? [y/n]')
        if is_final=='y':
            return True
        else:
            return False
    
    def print_final_price(self, price=None):
        if not price:
            selected_journey_id=int(input('Select Journey ID: '))
            final_price=self.df.iloc[selected_journey_id-1]['Price']
            print(f'{self.orig_df.iloc[selected_journey_id-1]}')
            print(f'Final price: {final_price}')
        else:
            # test code
            print(f'{self.orig_df.iloc[np.random.randint(1,1471,1)]}')
            print(f'Final price: {price}')         

    def get_standard_feature_dict(self, feature_dict):
        for k, v in feature_dict.items():
            feature_dict[k]=self.convert_to_ordinal(k, v)
        return feature_dict

    def convert_to_ordinal(self, feature, nom_val):
        if feature=='Region':
            return self.util_obj.regions.get(nom_val)
        elif feature=='Hotel':
            return self.util_obj.hotels.get(nom_val)
        elif feature=='Season':
            return nom_val-1
        else:
            return nom_val
    
    def adapt_case(self, user_pref):
        user_pref_list=[None]*len(self.predictors)
        for k,v in user_pref.items():
            user_pref_list[self.predictors.index(k)]=v
        # return self.reg.predict([user_pref_list])
        return 1000

if __name__=='__main__':
    input_file='data/travel_cb.txt'
    travel_cbr=ccbr(input_file)
    travel_cbr.start()