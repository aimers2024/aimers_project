import pandas as pd
import numpy as np
from category_encoders.ordinal import OrdinalEncoder
from pycaret.classification import *
from sklearn.model_selection import train_test_split
from preprocess import preprocess
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(22, 128)
    self.fc2 = nn.Linear(128, 256)
    self.fc3 = nn.Linear(256, 112)
    self.fc4 = nn.Linear(112, 56)
    self.fc5 = nn.Linear(56, 2)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc4(x)
    x = self.relu(x)
    x = self.dropout(x)
    x = self.fc5(x)
    return x

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

if __name__ == "__main__" : 
    wandb.login()
    wandb.init(project="automl")
    
    # 데이터 읽기
    df_train = pd.read_csv("train.csv")  # 학습용 데이터
    df_test = pd.read_csv("submission.csv")  # 테스트 데이터(제출파일의 데이터)
    df_test_id = df_test['id']
    df_test_no_id = df_test.drop(columns=['id'])
    # df_all = pd.concat([df_train, df_test_no_id]).reset_index(drop=True)
    # 데이터 전처리
    enc_ordinal = OrdinalEncoder()
    df_train, df_test = preprocess(df_train, df_test_no_id)
    df_train["is_converted"] = enc_ordinal.fit_transform(df_train["is_converted"])
    df_test["is_converted"] = enc_ordinal.transform(df_test["is_converted"])
    
    # 수치형 데이터 스케일링
    numeric_col = ["bant_submit", "com_reg_ver_win_rate", "customer_idx", 
               "historical_existing_cnt", "id_strategic_ver", "it_strategic_ver", "idit_strategic_ver", 
               "lead_desc_length", "ver_cus", "ver_pro", "ver_win_rate_x", "ver_win_ratio_per_bu", 
               "lead_owner", "is_converted"]
    scaler = MinMaxScaler()
    df_train[numeric_col] = scaler.fit_transform(df_train[numeric_col])
    df_test[numeric_col] = scaler.transform(df_test[numeric_col])
    
    x_train, x_val, y_train, y_val = train_test_split(
        df_train.drop("is_converted", axis=1),
        df_train["is_converted"],
        test_size=0.1,
        shuffle=True,
        random_state=400,
    )

    x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
    x_test_tensor = torch.tensor(df_test.values, dtype=torch.float32)
    
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    
    dataset = TensorDataset(x_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MLP()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 학습
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 100
    for epoch in range(1, num_epochs+1):
        total_loss = 0.0
        total_accuracy = 0.0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            accuracy = calculate_accuracy(outputs, batch_y)
            total_accuracy += accuracy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = total_accuracy / len(dataloader)

        if epoch % 10 == 0:
            print('Epoch [{}/{}], Loss: {}, Accuracy: {}'.format(
            epoch, num_epochs, avg_loss, avg_accuracy
        ))
    
    # eval
    model.eval()

    with torch.no_grad():
        x_test_tensor = x_test_tensor.to(device)

        predictions = model(x_test_tensor)

    _, predicted_labels = torch.max(predictions, 1)

    predicted_labels = predicted_labels.cpu().numpy()
    predicted_labels = enc_ordinal.inverse_transform(predicted_labels)
    
    # 제출 데이터 읽어오기 (df_test는 전처리된 데이터가 저장됨)
    df_sub = pd.read_csv("submission.csv")
    df_sub["is_converted"] = predicted_labels
    # 제출 파일 저장
    df_sub.to_csv("submission.csv", index=False)