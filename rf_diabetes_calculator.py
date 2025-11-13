import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ---------------------- 1. 加载训练好的RF模型 ----------------------
@st.cache_resource
def load_model():
    model_path = "训练好的模型/rf_model.pkl"
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("未找到模型文件！请先运行「二分类机器学习及可解释性.py」生成 rf_model.pkl")
        st.stop()

rf_model = load_model()

# ---------------------- 2. 特征列表（与模型训练时完全一致：顺序+名称） ----------------------
feature_names = [
    "gender", "exercise", "race", "his", "hyperlip", "pregnant",
    "age", "glucose", "pressure", "triceps", "bmi", "pedigree", "insulin"
]

# ---------------------- 3. 网页交互界面 ----------------------
st.title("糖尿病风险预测工具（基于随机森林模型）")
st.markdown("### 输入以下临床指标，点击「预测」获取风险评估")
st.markdown("注：所有输入需与临床检测数据一致，结果仅供参考，不替代医疗诊断")

input_features = {}

for feat in feature_names:
    # 分类变量（按模型特征顺序排列）
    if feat == "gender":
        input_features[feat] = st.selectbox(
            "性别", 
            options=[0, 1], 
            format_func=lambda x: "女" if x == 0 else "男",
            help="性别编码：女=0，男=1"
        )
    elif feat == "exercise":
        input_features[feat] = st.selectbox(
            "是否规律运动（每周≥3次）", 
            options=[0, 1], 
            format_func=lambda x: "否" if x == 0 else "是"
        )
    elif feat == "race":
        input_features[feat] = st.selectbox(
            "种族", 
            options=[0, 1, 2], 
            format_func=lambda x: {0: "亚洲人", 1: "白人", 2: "黑人"}[x]
        )
    elif feat == "his":
        input_features[feat] = st.selectbox(
            "是否有高血压病史", 
            options=[0, 1], 
            format_func=lambda x: "否" if x == 0 else "是"
        )
    elif feat == "hyperlip":
        input_features[feat] = st.selectbox(
            "是否有高血脂病史", 
            options=[0, 1], 
            format_func=lambda x: "否" if x == 0 else "是"
        )
    elif feat == "pregnant":
        input_features[feat] = st.selectbox(
            "是否有妊娠糖尿病史（女性适用）", 
            options=[0, 1], 
            format_func=lambda x: "否" if x == 0 else "是"
        )
    
    # 数值型特征（按模型特征顺序排列）
    elif feat == "age":
        input_features[feat] = st.number_input(
            "年龄（岁）", 
            min_value=18, max_value=100, value=50, step=1,
            help="请输入实际年龄，范围18-100岁"
        )
    elif feat == "glucose":
        input_features[feat] = st.number_input(
            "空腹血糖（mmol/L）", 
            min_value=3.0, max_value=30.0, value=5.5, step=0.1,
            help="正常范围3.9-6.1 mmol/L，超过7.0 mmol/L需警惕"
        )
    elif feat == "pressure":
        input_features[feat] = st.number_input(
            "收缩压（mmHg）", 
            min_value=80.0, max_value=200.0, value=120.0, step=1.0,
            help="正常范围90-139 mmHg，超过140 mmHg为高血压"
        )
    elif feat == "triceps":
        input_features[feat] = st.number_input(
            "三头肌皮褶厚度（mm）", 
            min_value=5.0, max_value=100.0, value=20.0, step=0.5,
            help="反映体脂率，成人正常范围10-40 mm"
        )
    elif feat == "bmi":
        input_features[feat] = st.number_input(
            "体重指数（BMI）", 
            min_value=10.0, max_value=60.0, value=25.0, step=0.1,
            help="正常范围18.5-23.9，24.0-27.9为超重，≥28.0为肥胖"
        )
    elif feat == "pedigree":
        input_features[feat] = st.number_input(
            "糖尿病家族史系数", 
            min_value=0.0, max_value=2.0, value=0.5, step=0.01,
            help="无家族史≈0.0-0.3，有直系亲属患病≈0.5-1.0"
        )
    elif feat == "insulin":
        input_features[feat] = st.number_input(
            "空腹胰岛素（mIU/L）", 
            min_value=0.0, max_value=300.0, value=50.0, step=1.0,
            help="正常范围5-25 mIU/L，过高可能提示胰岛素抵抗"
        )

# ---------------------- 4. 预测逻辑（强制特征顺序匹配） ----------------------
if st.button("开始预测", type="primary"):
    # 关键修复：按模型要求的特征顺序构建DataFrame，确保列顺序完全匹配
    input_df = pd.DataFrame([input_features])[feature_names]
    
    # 模型预测
    pred_prob = rf_model.predict_proba(input_df)[0, 1]
    
    # 结果展示
    st.markdown("---")
    st.markdown(f"### 糖尿病风险预测结果")
    st.markdown(f"**风险概率：{pred_prob:.2%}**")
    
    if pred_prob >= 0.7:
        st.warning("⚠️ 高风险：糖尿病发生概率≥70%")
        st.markdown("建议：立即就医进行糖耐量试验、糖化血红蛋白检测，调整饮食结构，增加运动频率，定期监测血糖。")
    elif pred_prob >= 0.3:
        st.info("🔍 中风险：糖尿病发生概率30%-69%")
        st.markdown("建议：每3个月监测一次血糖，控制体重（BMI<24），减少高糖高脂食物摄入，每周运动≥150分钟。")
    else:
        st.success("✅ 低风险：糖尿病发生概率<30%")
        st.markdown("建议：保持健康生活方式，每年进行一次体检，监测血糖、血压变化。")

# ---------------------- 5. 特征重要性可视化 ----------------------
st.markdown("---")
st.markdown("### 模型关键影响因素（特征重要性）")
feat_importance = pd.DataFrame({
    "临床指标": rf_model.feature_names_in_,
    "影响权重": rf_model.feature_importances_
}).sort_values("影响权重", ascending=False).head(8)

st.bar_chart(feat_importance.set_index("临床指标"), y="影响权重", height=300)
st.markdown("注：影响权重越高，对预测结果的贡献越大（基于随机森林模型计算）")