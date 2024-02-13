import os
from category_encoders import OneHotEncoder, OrdinalEncoder

dir_name = os.path.dirname(os.path.abspath(__file__)) # 현재 파일의 디렉토리 경로
class DataPreprocessor:
    def __init__(self, preprocess_columns):
        os.makedirs(os.path.join(dir_name, 'config'), exist_ok=True)
        self.preprocess_columns = preprocess_columns
        self.enc_ordinal = None
        self.frequency_mappings = {}
        self.enc_onehot = None

    def preprocess(self,df):
        #Preprocessing
        ##id_strategic_ver,it_strategic_ver 병합
        df['id_strategic_ver'] = df['id_strategic_ver'].fillna(0)
        df['it_strategic_ver'] = df['it_strategic_ver'].fillna(0)
        df['idit_strategic_ver'] = df['id_strategic_ver'] + df['it_strategic_ver']

        ##product_category가 null인 값 중 product_subcategory 값이 null이 아니면 해당 값 가져오기
        df.loc[df['product_category'].isnull() & df['product_subcategory'].notnull(), 'product_category'] = df['product_subcategory']

        #customer contry 열에서 나라만 뽑아보기
        df['customer_country'] = df["customer_country"].str.split('/').str[2]

        #Drop
        df = df.drop(self.preprocess_columns['drop_column'], axis=1)

        #Fill-Null
        for column in self.preprocess_columns['null_integer']:
            df[column] = df[column].fillna(0)
            
        for column in self.preprocess_columns['null_string']:
            df[column] = df[column].fillna('others')

        #Group
        for column in self.preprocess_columns['Grouping']:
            df.loc[df[column].isin(self.preprocess_columns["grouping_element"]), column] = 'others'

        ##value counts가 1인 값들 전부다 others로 바꾸기
        for column in self.preprocess_columns['Categorical']:
            value_counts = df[column].value_counts()

            ###빈도수가 1인 값들 찾기
            values_to_replace = value_counts[value_counts == 1].index.tolist()

            ###빈도수가 1인 값들을 'others'로 치환
            df[column] = df[column].replace(values_to_replace, 'others')
        return df
    
    def fit_transform(self, df):
        df = self.preprocess(df)
        # Ordinal encoding
        self.enc_ordinal = OrdinalEncoder(cols=self.preprocess_columns['Ordinal'])
        df = self.enc_ordinal.fit_transform(df)
       
        # Frequency encoding
        for column in self.preprocess_columns['Frequency']:
            frequency = df.groupby(column).size() / len(df)
            self.frequency_mappings[column] = frequency
            df[column] = df[column].apply(lambda x: self.frequency_mappings[column].get(x, 0))
        
        # One-hot encoding
        self.enc_onehot = OneHotEncoder(cols=self.preprocess_columns['OneHot'], use_cat_names=True)
        df = self.enc_onehot.fit_transform(df)

        return df

    def transform(self, df):
        df = self.preprocess(df)
        
        # 인코더 로드는 __init__에서 수행됨
        
        # Ordinal encoding
        df = self.enc_ordinal.transform(df)
        
        # Frequency encoding
        for column in self.preprocess_columns['Frequency']:
            df[column] = df[column].apply(lambda x: self.frequency_mappings[column].get(x, 0))
        
        # One-hot encoding
        df = self.enc_onehot.transform(df)
        
        return df
