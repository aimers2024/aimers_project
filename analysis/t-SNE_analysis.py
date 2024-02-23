import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def label_encoding(series: pd.Series) -> pd.Series:
    """범주형 데이터를 시리즈 형태로 받아 숫자형 데이터로 변환합니다."""

    my_dict = {}

    # 모든 요소를 문자열로 변환
    series = series.astype(str)

    for idx, value in enumerate(sorted(series.unique())):
        my_dict[value] = idx
    series = series.map(my_dict)

    return series

df_train = pd.read_csv("../data/train.csv")  # 학습용 데이터
df_test = pd.read_csv("../data/submission.csv")  # 테스트 데이터(제출파일의 데이터)

# 레이블 인코딩할 칼럼들
label_columns = [
    "customer_country",
    "business_subarea",
    "business_area",
    "business_unit",
    "customer_type",
    "enterprise",
    "customer_job",
    "inquiry_type",
    "product_category",
    "product_subcategory",
    "product_modelname",
    "customer_country.1",
    "customer_position",
    "response_corporate",
    "expected_timeline",
]


for col in label_columns:
    df_train[col] = label_encoding(df_train[col])

# 상관계수가 NaN인 id_strategic_ver, it_strategic_ver, idit_strategic_ver을 제거
col_to_drop = ['id_strategic_ver', 'it_strategic_ver', 'idit_strategic_ver']
df_train = df_train.drop(col_to_drop, axis=1)

# 결측치가 있는 행을 제거할 경우 1696개로 축소됨
df_train_filtered = df_train.dropna()
df_train_filtered.isnull()
# is_converted 열을 기준으로 TRUE와 FALSE로 나누기
df_true = df_train_filtered[df_train_filtered['is_converted'] == True]
df_false = df_train_filtered[df_train_filtered['is_converted'] == False]
print(f'True 개수: {df_true.shape[0]}')
print(f'False 개수: {df_false.shape[0]}')

tsne = TSNE(n_components=3, random_state=42, perplexity=100, n_iter=300)
tsne_result = tsne.fit_transform(df_train_filtered)

# 'is_converted' 값에 따라 데이터를 분할
tsne_result_true = tsne_result[df_train_filtered['is_converted'] == True]
tsne_result_false = tsne_result[df_train_filtered['is_converted'] == False]

# 3차원 산점도 그리기
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# TRUE 데이터 산점도 그리기
ax.scatter(tsne_result_true[:, 0], tsne_result_true[:, 1], tsne_result_true[:, 2], 
           label='is_converted: TRUE', c='blue', marker='o', alpha=0.5)

tsne_result_false = tsne_result_false[:157, :]
# FALSE 데이터 산점도 그리기
ax.scatter(tsne_result_false[:, 0], tsne_result_false[:, 1], tsne_result_false[:, 2], 
           label='is_converted: FALSE', c='red', marker='^', alpha=0.5)
# 그래프 설정
ax.set_title("t-SNE 3D Scatter Plot")
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.legend()

# 그래프 출력
plt.show()