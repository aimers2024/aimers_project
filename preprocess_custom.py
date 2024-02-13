import pandas as pd
import json
from category_encoders import OneHotEncoder, OrdinalEncoder

def preprocess(df_train, df_test) : 
    
    with open('preprocess.json') as f:
        preprocess_columns = json.load(f)
    df_list = [df_train, df_test]
    pdf_list = []
    ##Ordinal encoding
    enc_ordinal = OrdinalEncoder(cols = preprocess_columns['Ordinal'])
    ##One-hot encoding
    enc_onehot = OneHotEncoder(cols =preprocess_columns['OneHot'], use_cat_names = True)
    
    # preprocess_list = []
    for idx, df in enumerate(df_list) :
        #Preprocessing
        ##id_strategic_ver,it_strategic_ver 병합
        df['id_strategic_ver'] = df['id_strategic_ver'].fillna(0)
        df['it_strategic_ver'] = df['it_strategic_ver'].fillna(0)
        df['idit_strategic_ver'] = df['id_strategic_ver'] + df['it_strategic_ver']

        ##product_category가 null인 값 중 product_subcategory 값이 null이 아니면 해당 값 가져오기
        df.loc[df['product_category'].isnull() & df['product_subcategory'].notnull(), 'product_category'] = df['product_subcategory']

        #Drop
        df = df.drop(preprocess_columns['drop_column'], axis=1)

        #Fill-Null
        for column in preprocess_columns['null_integer']:
            df[column] = df[column].fillna(0)
        for column in preprocess_columns['null_integer']:
            df[column] = df[column].fillna('others')
            

        #Group
        for column in preprocess_columns['Grouping']:
            df.loc[df[column].isin(['Etc.']), column] = 'others'

        ##value counts가 1인 값들 전부다 others로 바꾸기
        for column in preprocess_columns['Categorical']:
            value_counts = df[column].value_counts()

            ###빈도수가 1인 값들 찾기
            values_to_replace = value_counts[value_counts == 1].index.tolist()

            ###빈도수가 1인 값들을 'others'로 치환
            df[column] = df[column].replace(values_to_replace, 'others')

        #Encoding
        if idx == 0 :
            ##Ordinal encoding
            df = enc_ordinal.fit_transform(df)
            ##One-hot encoding
            df = enc_onehot.fit_transform(df)
        else :
            ##Ordinal encoding
            df = enc_ordinal.transform(df)
            ##One-hot encoding
            df = enc_onehot.transform(df)

        ##Frequency encoding
        for column in preprocess_columns['Frequency']:
            enc_fre = (df.groupby(column).size()) / len(df)
            df[column] = df[column].apply(lambda x : enc_fre[x])
        pdf_list.append(df)
    return pdf_list
