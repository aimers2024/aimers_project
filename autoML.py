import pandas as pd
import numpy as np
from category_encoders.ordinal import OrdinalEncoder
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from preprocess import preprocess
import wandb

if __name__ == "__main__" : 
    # wandb.login()
    # wandb.init(project="automl")
    
    # Data read
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("submission.csv")
    
    # 테스트셋으로부터 id 분리
    df_test_id = df_test['id']
    df_test_no_id = df_test.drop(columns=['id'])
    # 데이터 일괄 전처리를 위한 concat
    df_all = pd.concat([df_train, df_test_no_id]).reset_index(drop=True)
    # 데이터 전처리
    df_all = preprocess(df_all)
    # 훈련, 테스트 셋 분리
    df_train = df_all.iloc[:len(df_train), :]
    df_test = df_all.iloc[len(df_train):, :]

    x_train, x_val, y_train, y_val = train_test_split(
        df_train,
        df_train["is_converted"],
        test_size=0.2,
        shuffle=True,
        random_state=400,
    )
    
    ## 모델 정의
    model = setup(
        data = x_train,
        test_data = x_val,
        normalize_method='minmax', 
        target = "is_converted",
        fold = 15,
        n_jobs=1
        # log_experiment="wandb"
        # log_plots= True
        # log_profile=True,
        # log_data=True
    )

    top3_model = compare_models(
               round=4,
               sort="F1",
               n_select = 3)
    print(f"top3_model : {top3_model}")
    tuned_top3 = [tune_model(i) for i in top3_model]
    
    blended_model = blend_models(estimator_list = tuned_top3)
    final_model = finalize_model(blended_model)
    
    # 인덱스 재정렬
    df_test = df_test.reset_index(drop=True)
    
    # 테스트셋 예측
    predictions = []
    pred = predict_model(final_model, data=df_test)
    
    # 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
    df_sub = pd.read_csv("submission.csv")
    df_sub["is_converted"] = pred["prediction_label"].astype(bool)

    # 제출 파일 저장
    df_sub.to_csv("submission.csv", index=False)