import pandas as pd
import util
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from collections import OrderedDict
import os

os.system('cls' if os.name == 'nt' else 'clear')

class ccbr():
    def __init__(self, input_file, similarity_metric='euclidean', is_weighted=False) -> None:
        self.util_obj=util.Util()
        self.orig_df=pd.read_csv('data/travel_cb_orig.csv', encoding='utf-8')
        # print(f'{self.orig_df.iloc[id]}')
        self.df=self.util_obj.build_df(input_file)
        self.df.to_csv('data/travel_cb.csv', encoding='utf-8')
        
        self. similarity_metric=similarity_metric
        self.is_weighted=is_weighted
        self.predictors=[
            'HolidayType','NumberOfPersons','Region', 'Transportation', 'Duration', 'Season', 
            'Accommodation','Hotel'
        ]
        self.feature_value_range=self.find_feature_value_range(self.predictors)
        xtrain, xtest, ytrain, ytest = train_test_split(self.df[self.predictors],self.df['Price'], 
                                                  random_state=42, 
                                                  test_size=0.20, shuffle=True)
        self.reg = LinearRegression().fit(xtrain, ytrain)
        self.predictors_1=['NumberOfPersons', 'Duration', 'Season']
        self.reg_1 = LinearRegression().fit(self.df[self.predictors_1], self.df['Price'])
        y_pred_train = self.reg.predict(xtrain)
        y_pred_test = self.reg.predict(xtest)
        coef= np.abs(self.reg.coef_)
        score = coef/sum(coef)
        self.score_dict = {}
        score = list(coef/sum(coef))
        for i in range(len(self.predictors)):
            self.score_dict[self.predictors[i]] = score[i] 
        self.score_dict = {k: round(v,2) for k, v in sorted(self.score_dict.items(), key=lambda item: item[1],reverse=True)}
        self.running_feature_ranklist=list(self.score_dict.keys())

        self.nominal_features=['Region', 'Hotel']
        self.ordinal_features=['NumberOfPersons', 'Duration', 'Season']
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
        self.user_pref={}

    def start(self):
        while self.running_feature_ranklist:
            selected_feature=self.get_q_preference()
            selected_feature_val=self.get_feature_val(selected_feature)
            self.user_pref[selected_feature]=selected_feature_val
            print()
            print('user preference: ')
            print(self.user_pref)
            print()
            self.user_pref=self.get_standard_feature_dict(self.user_pref)
            best_case_ids, best_case_scores=self.get_similar_cases(
                self.user_pref, metric=self.similarity_metric, weights=self.is_weighted
            )
            is_final=self.check_case_with_user(best_case_ids, best_case_scores)
            if is_final:
                self.print_final_price()
                break
            else:
                pass
        if not is_final:
            print('user choices: ')
            print(self.user_pref)
            self.print_final_price(self.adapt_case(self.user_pref))
        
    def get_q_preference(self):
        q_scores=[self.score_dict.get(feature) for feature in self.running_feature_ranklist[:3]]
        q_scores/=sum(q_scores)
        for i in range(3):
            try:
                ques=self.f_q_map.get(self.running_feature_ranklist[i])
            except IndexError:
                break
            print(f'{i+1}: {ques} (Score: {round(q_scores[i],2)})')
        selected_feature_id=None
        while not selected_feature_id:
            try:
                print()
                temp=int(input('Select one of the above Qs (1, 2 or 3): '))
                if temp in range(1, len(self.running_feature_ranklist)+1):
                    selected_feature_id=temp
                else:
                    print('Select a valid Question number! Try again!')
            except ValueError:
                print('Select a valid Question number! Try again!')
        selected_feature_name=self.running_feature_ranklist[selected_feature_id-1]
        self.running_feature_ranklist.pop(selected_feature_id-1)
        # self.update_feature_ranking(selected_feature_name)
        return selected_feature_name

    def get_feature_val(self, selected_feature):
        if selected_feature not in self.nominal_features:
            print('\n')
            return int(input(self.f_q_map_detailed.get(selected_feature)))
        else:
            print('\n')
            return input(self.f_q_map.get(selected_feature))
    
    def get_similar_cases(self, user_pref, k=3, metric='euclidean', weights=False):
        weights_arr=np.ones(len(self.predictors))/len(self.predictors)
        if weights:
            weights_arr=np.asarray(list(self.score_dict.values()))/sum(self.score_dict.values())
        # retrieve indices of k similar cases
        if metric=='euclidean':
            euc_dist_arr=np.full(len(self.df), np.inf).reshape(-1,1)
            for count, val in enumerate(euc_dist_arr):
                temp_similarity_arr=[0]*len(self.predictors)
                for count_i, val_i in enumerate(self.predictors):
                    temp_similarity_arr[count_i]=self.find_feature_similarity(val_i, user_pref.get(val_i), self.df.at[count,val_i])
                    temp_similarity_arr[count_i]*=weights_arr[count_i]
                euc_dist_arr[count]=np.linalg.norm(temp_similarity_arr)
            euc_dist_arr=np.amax(euc_dist_arr, axis=1)
            return np.argsort(-euc_dist_arr)[:k], np.sort(euc_dist_arr)[-k:]
        else:
            similarity_arr=np.full(len(self.df), 0)
            for count, val in enumerate(similarity_arr):
                temp_similarity_arr=[0]*len(self.predictors)
                for count_i, val_i in enumerate(self.predictors):
                    temp_similarity_arr[count_i]=self.find_feature_similarity(val_i, user_pref.get(val_i), self.df.at[count,val_i])
                similarity_arr[count]=np.linalg.norm(temp_similarity_arr)
                similarity_arr=similarity_arr*weights_arr

            similarity_arr=similarity_arr.sum(axis=1)
            return np.mean(similarity_arr)[:k], np.sort(-euc_dist_arr)[:k]

    def find_feature_similarity(self, feature_name, q_feature, c_feature):
        if not q_feature:
            return 0
        elif feature_name not in self.ordinal_features:
            return 1 if q_feature==c_feature else 0
        elif feature_name=='Season':
            if q_feature<c_feature:
                s1=q_feature
                s2=c_feature
            else:
                s1=c_feature
                s2=q_feature
            return 1-(min((s2-s1),(s1-s2+12))/self.feature_value_range.get(feature_name))
        else:
            return 1-(abs(q_feature-c_feature)/self.feature_value_range.get(feature_name))
    
    def find_feature_value_range(self, predictors):
        feature_value_range={key: 0 for key in predictors}
        for feature in predictors:
            feature_value_range[feature]=self.df[feature].max()-self.df[feature].min()
        return feature_value_range

    def check_case_with_user(self, best_case_ids, best_case_scores):
        best_case_scores/=sum(best_case_scores)
        for index, (val1, val2) in enumerate(zip(best_case_ids, best_case_scores)):
            print('\n')
            print('Case Score: ', round(val2,2))
            print(f'{self.orig_df.iloc[val1, [0,1,3,4,5,6,7,8,9]].to_string()}')
        print('\n')
        is_final=input('Do you want to select a Journey? [y/n]')
        print()
        if is_final=='y':
            return True
        else:
            return False
    
    def print_final_price(self, price=None):
        if not price:
            selected_journey_code=int(input('Select Journey Code: '))
            print()
            final_price=self.df.iloc[selected_journey_code-1]['Price']
            print('Here are your travel details: ')
            print(f'{self.orig_df.iloc[selected_journey_code-1, [0,1,3,4,5,6,7,8,9]].to_string()}')
            print()
            print(f'Final price is: {final_price}')
            print()
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
            return int(nom_val)
    
    def adapt_case(self, user_pref):
        user_pref_list=[None]*len(self.predictors)
        for k,v in user_pref.items():
            user_pref_list[self.predictors.index(k)]=v
        return self.reg_1.predict([list(user_pref_list[i] for i in [1, 4, 5] )])[0]

if __name__=='__main__':
    input_file='data/travel_cb.txt'
    travel_cbr=ccbr(input_file, is_weighted=True)
    # travel_cbr.start()
    travel_cbr.start()