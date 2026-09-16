import os
import pandas as pd
import subprocess
import glob
from tqdm import tqdm

# sklearn은 머신러닝 모델 학습과 평가에 사용됩니다.
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# Non-VPN 및 VPN 데이터셋이 저장된 디렉토리 경로를 지정합니다.
NONVPN_DIR = "dataset/NonVPN-PCAPs-01"
VPN_DIR = "dataset/VPN-PCAPS-01"

# 추출된 CSV 파일들을 저장할 출력 디렉토리입니다.
OUTPUT_DIR = "dataset/output_csvs"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 디렉토리가 없으면 생성

# 파일 이름에 포함된 키워드를 바탕으로 트래픽 유형을 분류합니다.
def get_traffic_type(filename):
    fname = filename.lower()
    if 'chat' in fname:
        return 'chat'
    elif 'email' in fname:
        return 'email'
    elif 'audio' in fname:
        return 'audio'
    elif 'video' in fname:
        return 'video'
    elif 'ftp' in fname or 'file' in fname or 'bittorrent' in fname:
        return 'file_transfer'
    else:
        return 'other'  # 위의 조건에 포함되지 않으면 기타로 분류
    
# tshark를 사용하여 pcap 파일로부터 주요 통계 피처를 추출한 후, CSV로 저장합니다.
def extract_and_save_csv(pcap_path, output_csv_path, label):
    tshark_fields = [
        "-T", "fields",
        "-e", "frame.time_relative",  # 패킷의 상대 시간
        "-e", "frame.len",            # 패킷 길이
        "-e", "ip.src",               # 출발지 IP
        "-e", "ip.dst",               # 목적지 IP
        "-e", "ip.proto",               # 프로토콜
        "-e", "tcp.srcport",               # 출발지 TCP PORT
        "-e", "tcp.dstport",               # 목적지 TCP PORT
        "-e", "udp.srcport",               # 출발지 UDP PORT
        "-e", "udp.dstport",               # 목적지 UDP PORT
        "-E", "separator=,"
    ]
    cmd = ["tshark", "-r", pcap_path] + tshark_fields

    try:
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
        # print(output)
        lines = output.strip().split("\n")
        packets = [line.split(",") for line in lines]
        df = pd.DataFrame(packets, columns=["time", "length", "src", "dst", "protocol", "tcp_src_port", "tcp_dst_port", "udp_src_port", "udp_dst_port"])
        df["time"] = pd.to_numeric(df["time"], errors='coerce')
        df["length"] = pd.to_numeric(df["length"], errors='coerce')
        df = df.dropna()

        # 흐름 단위 정의 (src-IP와 dst-IP 조합)
        df["flow"] = df["src"] + "-" + df["dst"]

        # 흐름 기준으로 통계 피처 집계
        features = df.groupby("flow").agg({
            "time": ["min", "max"],
            "length": ["count", "sum", "mean", "std"]
        }).reset_index()

        # 컬럼 이름 정리
        features.columns = ["flow", "time_min", "time_max", "packet_count", "total_bytes", "avg_pkt_size", "std_pkt_size", "protocol", "tcp_src_port", "tcp_dst_port", "udp_src_port", "udp_dst_port"]
        features["duration"] = features["time_max"] - features["time_min"]  # 흐름 지속 시간 계산
        features = features.drop(columns=["time_min", "time_max"])  # 불필요한 컬럼 제거
        features["label"] = label  # 클래스 레이블 추가

        # 결과를 CSV로 저장
        if not features.empty:
            features.to_csv(output_csv_path, index=False)
    except Exception as e:
        print(f"[⚠️] Error processing {pcap_path}: {e}")
        
# 전체 pcap 파일들을 순회하며 피처를 추출하고 CSV 파일로 저장합니다.
def batch_convert_pcaps():
    for folder in [NONVPN_DIR, VPN_DIR]:
        pcap_files = glob.glob(os.path.join(folder, "*.pcap")) + glob.glob(os.path.join(folder, "*.pcapng"))
        for pcap_file in tqdm(pcap_files, desc=f"Processing {folder}"):
            filename = os.path.basename(pcap_file)
            label = get_traffic_type(filename)
            output_csv_path = os.path.join(OUTPUT_DIR, filename.replace(".pcapng", ".csv").replace(".pcap", ".csv"))
            extract_and_save_csv(pcap_file, output_csv_path, label)

# 개별 CSV 파일들을 하나로 합치고, 데이터를 셔플하여 저장합니다.
INPUT_DIR = OUTPUT_DIR
OUTPUT_CSV = "output_multiple-1.csv"

def merge_all_csvs():
    all_csvs = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    dfs = []
    for csv_file in all_csvs:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"[⚠️] Error reading {csv_file}: {e}")

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df = shuffle(final_df, random_state=42)  # 무작위 섞기
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Merged CSV saved to {OUTPUT_CSV} (Total: {len(final_df)} rows)")
    else:
        print("❌ No CSV files found or all failed.")

def model_ml_train():
    # 학습용 데이터를 불러오고, 랜덤포레스트 모델을 학습한 후 평가합니다.
    df = pd.read_csv("output_multiple.csv").dropna()
    X = df.drop(columns=["flow", "label"])  # 피처만 사용
    y = df["label"]

    # 문자열 라벨을 숫자로 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 학습용/테스트용 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42)

    # 모델 정의 및 학습
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # 평가 지표 출력
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix 시각화
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    return clf

def model_dl_train(clf):
    # PyTorch를 사용한 MLP 기반 딥러닝 분류 모델입니다.
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    import seaborn as sns
    from sklearn.tree import plot_tree
    import matplotlib.pyplot as plt

    # 데이터 로드 및 전처리
    df = pd.read_csv("output_multiple.csv").dropna()
    X = df.drop(columns=["flow", "label"])
    y = df["label"]

    # 라벨 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # 특성 스케일링 (표준화)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tensor 변환
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    # 학습/테스트 분할
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.3, stratify=y_tensor, random_state=42
    )

    # 데이터 로더 준비
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # MLP 모델 정의
    class MLP(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(MLP, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.model(x)

    # 장치 설정 (GPU 사용 가능 시 활용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim=X.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    epochs = 200
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # 평가
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    # 분류 리포트 출력
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

    # 혼동 행렬 시각화
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Oranges")
    plt.title("PyTorch Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    # 학습된 랜덤포레스트 모델을 통해 각 피처의 중요도를 시각화합니다.

    # 피처 이름 목록
    feature_names = X.columns

    # 중요도 추출
    importances = clf.feature_importances_

    # 중요도를 기준으로 정렬
    indices = importances.argsort()[::-1]

    # 시각화
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names[indices])
    plt.title("Feature Importances from Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    
    # 랜덤포레스트 내부의 일부 트리를 시각화합니다.
    

    # 몇 개의 트리를 시각화할지 설정
    n_trees_to_plot = 1

    plt.figure(figsize=(20, 10))
    for i in range(n_trees_to_plot):
        plt.clf()
        estimator = clf.estimators_[i]
        plot_tree(estimator,
                feature_names=X.columns,
                class_names=le.classes_,
                filled=True,
                rounded=True,
                max_depth=3)  # 깊이를 제한하여 보기 좋게
        plt.title(f"Random Forest Tree #{i}")
        plt.tight_layout()
        plt.show()


# 변환 실행
if __name__ == "__main__":
    batch_convert_pcaps()
    # merge_all_csvs()
    # clf = model_ml_train()
    # model_dl_train(clf)