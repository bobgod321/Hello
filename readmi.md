# 🚢 Titanic Survival Prediction – Kaggle 海难生死预测

## 📌 项目背景

本项目基于 Kaggle 经典竞赛 **"Titanic: Machine Learning from Disaster"**，目标是预测泰坦尼克号沉船事件中乘客的生存情况。

1912年4月15日，泰坦尼克号在首航中与冰山相撞沉没，造成1502人遇难。通过分析乘客的个人信息（如性别、年龄、舱位等级、家庭关系等），构建分类模型，预测其是否生还。

---

## 🎯 问题定义

- **任务类型**：二分类（Binary Classification）
- **目标变量**：`Survived`（1 = 生还，0 = 遇难）
- **评估指标**：分类准确率（Accuracy）、ROC-AUC

---

## 📊 数据集说明

| 数据集 | 样本数 | 特征数 | 说明 |
|--------|--------|--------|------|
| 训练集 | 891 | 12 | 包含标签 `Survived` |
| 测试集 | 418 | 11 | 无标签，用于最终预测 |

### 原始特征

| 特征 | 类型 | 描述 |
|------|------|------|
| PassengerId | int | 乘客ID |
| Survived | int | 生存标签（0/1） |
| Pclass | int | 舱位等级（1/2/3） |
| Name | str | 乘客姓名 |
| Sex | str | 性别 |
| Age | float | 年龄 |
| SibSp | int | 兄弟姐妹/配偶数量 |
| Parch | int | 父母/子女数量 |
| Ticket | str | 船票号 |
| Fare | float | 船票价格 |
| Cabin | str | 船舱编号 |
| Embarked | str | 登船港口（C/Q/S） |

---

## 🧠 方法流程

### 1️⃣ 数据探索与可视化

- 分析各特征与生存率的关系：
  - 港口与生存率：C港口生存率最高（55.4%）
  - 票价与生存率：生还者票价分布更偏向高价位
  - 年龄分组：儿童生存率最高
  - 性别：女性生存率远高于男性
  - 舱位等级：头等舱生存率最高
- 使用条形图、KDE密度图、热力图等可视化工具

### 2️⃣ 数据预处理

#### 缺失值处理

| 特征 | 缺失数量 | 处理方式 |
|------|----------|----------|
| Age | 263 | 随机森林回归模型预测填充 |
| Cabin | 1014 | 填充为 `'U'`（Unknown） |
| Embarked | 2 | 填充为众数 `'S'` |
| Fare | 1 | 按 Pclass=3, Embarked='S' 的均值填充 |

#### 特征工程

| 新特征 | 构造方法 |
|--------|----------|
| Title | 从姓名中提取头衔（Mr/Mrs/Miss/Master等）并分组合并 |
| Family_num | Parch + SibSp + 1 |
| familySize | 根据 Family_num 分类：0=独行，1=中家庭(2-4人)，2=大家庭(≥5人) |
| Deck | 提取 Cabin 首字母 |
| TickCom | 统计同票号乘客数 |
| TickGroup | 根据 TickCom 分类：0=单人票，1=小团体(2-4人)，2=大团体(≥5人) |

#### 特征编码

- 对分类特征进行 one-hot 编码：
  - `Sex` → `Sex_female`, `Sex_male`
  - `Embarked` → `Embarked_C`, `Embarked_Q`, `Embarked_S`
  - `Title` → `Title_Master`, `Title_Miss`, `Title_Mr`, `Title_Mrs`, `Title_Officer`, `Title_Royalty`

### 3️⃣ 特征选择

最终保留的核心特征：

| 特征 | 说明 |
|------|------|
| Pclass | 舱位等级 |
| Fare | 票价（对数变换） |
| familySize | 家庭规模分类 |
| TickGroup | 票号乘客分组 |
| Sex_* | 性别编码 |
| Embarked_* | 港口编码 |
| Title_* | 头衔编码 |

### 4️⃣ 模型训练与调优

#### 候选模型

| 模型 | 说明 |
|------|------|
| SVC | 支持向量机 |
| Decision Tree | 决策树 |
| Random Forest | 随机森林 |
| Extra Trees | 极端随机树 |
| Gradient Boosting | 梯度提升树 |
| KNN | K近邻 |
| Logistic Regression | 逻辑回归 |
| LDA | 线性判别分析 |
| XGBoost | XGBoost |

#### 交叉验证

- 使用 `StratifiedKFold`，n_splits=10
- 评估指标：准确率（Accuracy）

#### 超参数调优

| 模型 | 调优方法 | 最佳参数 |
|------|----------|----------|
| SVC | GridSearchCV | C=0.5, gamma=0.1, kernel='rbf' |
| XGBoost | RandomizedSearchCV | 最优参数组合 |
| GradientBoosting | GridSearchCV | 最优参数组合 |
| LDA | GridSearchCV | solver='lsqr', shrinkage=None |

### 5️⃣ 模型集成

- 使用 **软投票（Soft Voting）** 融合：
  - GradientBoosting（权重 2）
  - XGBoost（权重 1）

---

## 📈 模型表现

### 交叉验证准确率

| 模型 | CV 平均准确率 | CV 标准差 |
|------|---------------|-----------|
| GradientBoosting | 0.8384 | 0.0441 |
| XGBoost | 0.8429 | 0.0468 |
| SVC | 0.8361 | 0.0353 |
| LDA | 0.8260 | 0.0322 |
| KNN | 0.8306 | 0.0416 |
| Random Forest | 0.8138 | 0.0329 |

### 测试集评估（内部验证）

| 模型 | 准确率 | 精确率 | 召回率 | 特异度 | F1分数 | AUC |
|------|--------|--------|--------|--------|--------|-----|
| GradientBoosting | 0.9080 | 0.9012 | 0.8538 | 0.9417 | 0.8769 | 0.95 |
| XGBoost | 0.8979 | 0.8984 | 0.8275 | 0.9417 | 0.8615 | 0.94 |
| SVC | 0.8373 | 0.8167 | 0.7427 | 0.8962 | 0.7779 | 0.91 |
| LDA | 0.8305 | 0.8051 | 0.7368 | 0.8889 | 0.7695 | 0.89 |

### 混淆矩阵（GradientBoosting）


| 模型 | 混淆矩阵 |
| ---- | -------- |
| GradientBoostingClassifier | [[517  32]<br> [ 50 292]] |
| XGBoost | [[517  32]<br> [ 59 283]] |
| SVC | [[492  57]<br> [ 88 254]] |
| LDA | [[488  61]<br> [ 90 252]] |

---

## ✅ 结论

1. **GradientBoosting 和 XGBoost 表现最佳**，在交叉验证和测试集上均取得较高准确率。
2. **特征工程对模型提升显著**，尤其是 `Title`、`familySize`、`TickGroup` 等构造特征有效捕捉了乘客的社会关系和群体特征。
3. **性别和舱位等级是最强预测因子**：女性生还率远高于男性，头等舱生还率最高。
4. **软投票集成**进一步提升了模型的稳定性和泛化能力。

---

## 📁 输出文件

| 文件 | 说明 |
|------|------|
| `Titanic_ensemble_Soft.csv` | 集成模型对测试集的预测结果（PassengerId + Survived） |

---

## 🛠️ 运行环境

```bash
Python 3.9+
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn