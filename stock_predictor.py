import pandas as pd
import numpy as np
import talib
import json
import yfinance as yf
import joblib
import os
from textblob import TextBlob  # 用于新闻情感分析
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import accuracy_score
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser

class StockPredictor:
    def __init__(self, ticker, start_date='2022-01-01', end_date='2023-12-31', model_path=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._get_historical_data()
        self._add_technical_indicators()
        self.features = self._create_features()
        
        if model_path and os.path.exists(model_path):
            self.model = self.load_model(model_path)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            
        self.history_file = "prediction_history.json"
        self.history_predictions = self._load_history()  # 初始化时加载历史记录
        
        # 如果历史文件不存在则初始化示例数据
        if not os.path.exists(self.history_file):
            # 添加初始示例记录
            self.history_predictions = [{
                'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'rise_prob': 50.0,
                'fall_prob': 50.0,
                'ticker': self.ticker,
                'close_price': self.data['Close'].iloc[-1] if not self.data.empty else 0.0,
                'model': 'Initial',
                'actual': None,
                'is_correct': None,
                'features': {k: 0.0 for k in self.features.columns if k != 'Target'}
            }]
            self._save_history()
            print("初始化创建历史记录文件并添加示例数据")
        
    def _get_historical_data(self):
        """获取历史股价数据"""
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=self.start_date, end=self.end_date)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    def _add_technical_indicators(self):
        """计算技术指标"""
        # MACD指标
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(
            self.data['Close'], 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9
        )
        
        # 识别金叉/死叉
        self.data['MACD_Cross'] = np.where(
            self.data['MACD'] > self.data['MACD_signal'], 1, 
            np.where(self.data['MACD'] < self.data['MACD_signal'], -1, 0)
        )
        
        # 成交量变化率
        self.data['Volume_change'] = self.data['Volume'].pct_change()
        
        # 布林带指标
        self.data['BB_Upper'], self.data['BB_Middle'], self.data['BB_Lower'] = talib.BBANDS(
            self.data['Close'],
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        self.data['Close_vs_BB_Upper'] = self.data['Close'] / self.data['BB_Upper'] - 1
        self.data['Close_vs_BB_Lower'] = self.data['Close'] / self.data['BB_Lower'] - 1
        
        # RSI指标
        self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)
        
        # 平均真实波幅(ATR)
        self.data['ATR'] = talib.ATR(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'], 
            timeperiod=14
        )
        
        # 动量指标
        self.data['Momentum'] = talib.MOM(self.data['Close'], timeperiod=10)
        
        # 能量潮指标(OBV)
        self.data['OBV'] = talib.OBV(self.data['Close'], self.data['Volume'])
        
    def _get_news_sentiment(self):
        """获取新闻情感分析（模拟数据）"""
        # 实际应用时应接入新闻API
        fake_news = [
            "Company reports strong earnings growth",
            "Industry faces regulatory challenges",
            "New product launch announced"
        ]
        sentiments = [TextBlob(text).sentiment.polarity for text in fake_news]
        return np.mean(sentiments)
    
    def _create_features(self):
        """创建特征数据集"""
        features = pd.DataFrame({
            'MACD_hist': self.data['MACD_hist'],
            'RSI': self.data['RSI'],
            'Volume_change': self.data['Volume_change'],
            '3_day_ma': talib.MA(self.data['Close'], timeperiod=3),
            '10_day_ma': talib.MA(self.data['Close'], timeperiod=10),
            'MACD_Cross': self.data['MACD_Cross'],
            'BB_Width': self.data['BB_Width'],
            'Close_vs_BB_Upper': self.data['Close_vs_BB_Upper'],
            'Close_vs_BB_Lower': self.data['Close_vs_BB_Lower'],
            
            # 新增技术指标
            'WILLR': talib.WILLR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14),  # 威廉指标
            'CCI': talib.CCI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=20),  # 商品通道指标
            'Volatility': self.data['Close'].rolling(window=10).std().pct_change(),  # 价格波动率
            'Volume_Price_Corr': self.data['Volume'].rolling(window=5).corr(self.data['Close']),  # 量价相关性
            'Price_Dispersion': (self.data['High'].rolling(window=5).max() - self.data['Low'].rolling(window=5).min()) / self.data['Close'],  # 价格离散度
            'MA_Slope': talib.LINEARREG_SLOPE(talib.MA(self.data['Close'], timeperiod=5), timeperiod=5),  # 移动平均线斜率
            
            # 新增技术指标组合特征
            'MACD_RSI_Combined': self.data['MACD_hist'] * self.data['RSI'],  # MACD与RSI组合
            'BB_Volume_Interaction': self.data['BB_Width'] * self.data['Volume_change'],  # 布林带宽度与成交量交互
            'Momentum_ATR_Ratio': self.data['Momentum'] / self.data['ATR'],  # 动量与波动率比值
            
            # 新增统计特征
            '3D_Return_Volatility': self.data['Close'].pct_change().rolling(3).std(),  # 3日收益率波动率
            '5D_Return_Volatility': self.data['Close'].pct_change().rolling(5).std(),  # 5日收益率波动率
            'ATR_5MA': self.data['ATR'].rolling(5).mean(),  # ATR5日移动平均
            'OBV_3MA': self.data['OBV'].rolling(3).mean()  # OBV3日移动平均
        })
        
        # 添加滞后特征
        for lag in [1, 2, 3]:
            features[f'Close_lag_{lag}'] = self.data['Close'].shift(lag)
            
        # 添加新闻情感特征
        features['News_Sentiment'] = self._get_news_sentiment()
        
        # 添加目标变量（次日涨跌）
        features['Target'] = np.where(
            self.data['Close'].shift(-1) > self.data['Close'], 1, 0
        )

        # 数据清洗
        # 1. 替换无穷值为NaN
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 2. 删除包含NaN的行
        features.dropna(inplace=True)
        # 3. 添加微小值防止除以零
        eps = 1e-6
        features['Price_Dispersion'] = features['Price_Dispersion'].add(eps)
        features['Momentum_ATR_Ratio'] = features['Momentum_ATR_Ratio'].add(eps)
        
        return features
    
    def _load_history(self):
        """从文件加载历史预测记录"""
        try:
            if not os.path.exists(self.history_file):
                return []
                
            with open(self.history_file, 'r') as f:
                history = json.load(f)
                
            # 数据格式验证和修复
            valid_history = []
            for record in history:
                # 检查必要字段
                if all(key in record for key in ['date', 'rise_prob', 'close_price']):
                    # 转换日期格式为字符串（如果存储的是datetime）
                    if isinstance(record['date'], dict):  # 处理datetime对象序列化后的字典
                        try:
                            record['date'] = datetime.datetime(**record['date']).strftime('%Y-%m-%d')
                        except:
                            record['date'] = datetime.datetime.now().strftime('%Y-%m-%d')
                    valid_history.append(record)
                    
            print(f"成功加载 {len(valid_history)} 条历史记录")
            return valid_history
            
        except json.JSONDecodeError:
            print("历史记录文件格式错误，将使用空记录")
            return []
        except PermissionError:
            print("没有文件访问权限，将使用空记录")
            return []
        except Exception as e:
            print(f"加载历史记录失败: {str(e)}")
            return []

    def _save_history(self):
        """保存历史预测记录到文件"""
        try:
            # 转换日期格式为字符串
            for record in self.history_predictions:
                if 'date' in record:
                    if isinstance(record['date'], datetime.datetime):
                        record['date'] = record['date'].strftime('%Y-%m-%d')
                    elif isinstance(record['date'], dict):  # 处理可能的序列化残留
                        record['date'] = datetime.datetime(**record['date']).strftime('%Y-%m-%d')

            # 简化保存流程
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.history_predictions,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    default=lambda o: o.isoformat() if isinstance(o, datetime.datetime) else str(o)
                )
            
            print(f"成功保存 {len(self.history_predictions)} 条记录到 {os.path.abspath(self.history_file)}")

        except Exception as e:
            print(f"保存失败: {str(e)}")
            # 确保至少保存空列表
            try:
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            except:
                pass

    def update_actual_prices(self):
        """更新最近预测的实际结果"""
        try:
            # 获取最新行情数据（比预测日期晚一天）
            last_pred_date = datetime.datetime.strptime(self.history_predictions[-1]['date'], '%Y-%m-%d')
            update_date = (last_pred_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            new_data = yf.Ticker(self.ticker).history(start=update_date, end=update_date)
            
            if not new_data.empty:
                actual_close = new_data['Close'].iloc[0]
                # 更新最近未更新的预测记录
                for pred in reversed(self.history_predictions):
                    if pred['actual'] is None:
                        pred['actual'] = actual_close
                        pred['is_correct'] = (actual_close > pred['close_price']) == (pred['rise_prob'] > 50)
                self._save_history()
        except Exception as e:
            print(f"更新实际价格失败: {str(e)}")

    def train_model(self, use_history=True):
        """训练并优化预测模型"""
        # 合并历史数据
        if use_history and len(self.history_predictions) > 10:
            historical_features = pd.DataFrame([p['features'] for p in self.history_predictions])
            historical_target = pd.Series([
                int(p['actual'] > p['close_price']) for p in self.history_predictions 
                if p['actual'] is not None
            ])
            
            # 合并数据集
            X = pd.concat([self.features.drop('Target', axis=1), historical_features])
            y = pd.concat([self.features['Target'], historical_target])
        else:
            X = self.features.drop('Target', axis=1)
            y = self.features['Target']
        
        # 特征重要性分析
        selector = RandomForestClassifier(n_estimators=100, random_state=42)
        selector.fit(X, y)
        
        # 获取特征重要性并选择前20个重要特征
        feature_importances = pd.Series(selector.feature_importances_, index=X.columns)
        selected_features = feature_importances.nlargest(20).index.tolist()
        X_selected = X[selected_features]
        
        # 初始化多个分类器（增加特征选择步骤）
        models = {
            'RandomForest': Pipeline([
                ('selector', ColumnTransformer([('select', 'passthrough', selected_features)])),
                ('model', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42))
            ]),
            'GradientBoosting': Pipeline([
                ('selector', ColumnTransformer([('select', 'passthrough', selected_features)])),
                ('model', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3))
            ]),
            'LogisticRegression': Pipeline([
                ('selector', ColumnTransformer([('select', 'passthrough', selected_features)])),
                ('scaler', StandardScaler()),
                ('model', LogisticRegression(C=0.1, solver='lbfgs', max_iter=1000))
            ]),
            'SVM': Pipeline([
                ('selector', ColumnTransformer([('select', 'passthrough', selected_features)])),
                ('scaler', StandardScaler()),
                ('model', SVC(C=1.0, kernel='rbf', probability=True))
            ]),
            'XGBoost': Pipeline([
                ('selector', ColumnTransformer([('select', 'passthrough', selected_features)])),
                ('model', XGBClassifier(
                    n_estimators=300, 
                    learning_rate=0.05, 
                    max_depth=4,
                    early_stopping_rounds=None,  # 显式禁用早停
                    enable_categorical=False,  # 明确关闭分类特征支持
                    eval_metric=None,  # 完全禁用评估指标计算
                    validate_parameters=False,  # 禁用参数验证检查
                    verbosity=0,  # 完全禁用日志输出
                    check_integrity=False,  # 跳过数据完整性检查
                    callbacks=[]  # 禁用所有回调函数
                ))
            ])
        }

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        best_score = 0
        self.best_model = None
        
        print("开始模型训练与评估...")
        for name, model in models.items():
            scores = []
            auc_scores = []
            
            # 执行时间序列交叉验证
            for train_index, test_index in tscv.split(X_selected):
                X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                # 添加验证集用于早停
                model.fit(X_train, y_train)
                
                preds = model.predict(X_test)
                proba = model.predict_proba(X_test)[:, 1]
                
                scores.append(accuracy_score(y_test, preds))
                auc_scores.append(roc_auc_score(y_test, proba))
            
            # 计算平均表现
            mean_acc = np.mean(scores)
            mean_auc = np.mean(auc_scores)
            
            # 保存最佳模型
            if mean_auc > best_score:
                best_score = mean_auc
                self.best_model = model
                # 保存特征选择器供后续使用
                self.feature_selector = model.named_steps['selector']
                # 保存最终选择的特征列表
                self.selected_features = selected_features
                # 保存完整模型管道
                joblib.dump(model, 'best_model.pkl')
                print(f"  当前最佳模型已保存：{name}")
            
            print(f"{name}模型评估报告：")
            print(f"  平均准确率：{mean_acc:.2%}")
            print(f"  平均AUC：{mean_auc:.2%}")
            print(f"  准确率标准差：{np.std(scores):.2%}")
            print(f"  AUC标准差：{np.std(auc_scores):.2%}")
        
        # 训练完成后自动保存历史记录
        self._save_history()
    def save_model(self, filename='best_model.pkl'):
        """保存训练好的模型到文件"""
        if self.best_model is None:
            raise ValueError("尚未训练模型，无法保存")
        joblib.dump(self.best_model, filename)
        print(f"模型已保存至 {os.path.abspath(filename)}")

    def load_model(self, filename='best_model.pkl'):
        """从文件加载训练好的模型"""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"模型文件 {filename} 不存在")
        self.best_model = joblib.load(filename)
        print(f"已从 {os.path.abspath(filename)} 加载模型")

    def predict_probability(self):
        """使用最佳模型预测涨跌概率"""
        if self.best_model is None:
            raise ValueError("尚未训练模型，请先执行train_model方法")
        
        raw_features = self.features.drop('Target', axis=1).iloc[-1:]
        transformed_features = self.feature_selector.transform(raw_features)
        latest_features = pd.DataFrame(transformed_features, 
                                     columns=self.selected_features,
                                     index=raw_features.index)
        
        proba = self.best_model.predict_proba(latest_features)[0]
        
        # 记录预测结果
        prediction = {
            'rise_prob': proba[1] * 100,
            'fall_prob': proba[0] * 100,
            'ticker': self.ticker,
            'close_price': self.data['Close'].iloc[-1],
            'date': self.data.index[-1].strftime('%Y-%m-%d'),
            'model': type(self.best_model).__name__,
            'actual': None  # 次日更新实际结果
        }
        
        # 保存历史预测
        if len(self.history_predictions) >= 30:  # 保留最近30天记录
            self.history_predictions.pop(0)
        self.history_predictions.append(prediction)
        
        return prediction

    def _get_macd_trend(self):
        """分析MACD趋势"""
        last_3 = self.data['MACD_hist'][-3:].values
        if all(x > 0 for x in last_3) and last_3[-1] > last_3[0]:
            return "强势上涨"
        elif all(x < 0 for x in last_3) and last_3[-1] < last_3[0]:
            return "强势下跌"
        return "震荡整理"

    def _get_rsi_status(self):
        """分析RSI状态"""
        rsi = self.data['RSI'].iloc[-1]
        if rsi > 70:
            return "超买区域"
        elif rsi < 30:
            return "超卖区域"
        return "正常区间"

    def _get_recent_accuracy(self):
        """计算最近5次预测准确率"""
        if len(self.history_predictions) < 2:
            return "N/A"
        
        correct = 0
        for i in range(1, min(5, len(self.history_predictions))):
            pred = self.history_predictions[-i-1]
            actual_rise = self.data['Close'].iloc[-i] > pred['close_price']
            correct += 1 if (pred['rise_prob'] > 50) == actual_rise else 0
            
        return round(correct / (min(5, len(self.history_predictions)-1)) * 100, 1)

    def _get_overall_accuracy(self):
        """计算总体预测准确率"""
        if len(self.history_predictions) < 2:
            return 0
            
        correct = sum(1 for p in self.history_predictions 
                     if 'is_correct' in p and p['is_correct'])
        
        return round(correct / len(self.history_predictions) * 100, 1) if self.history_predictions else 0

    def _prepare_retraining_data(self):
        """准备再训练数据集"""
        # 从历史预测中提取特征
        historical_features = pd.DataFrame([p['features'] for p in self.history_predictions])
        historical_target = pd.Series([
            (p['actual'] > p['close_price']) for p in self.history_predictions 
            if p['actual'] is not None
        ]).astype(int)
        
        # 创建时间序列特征
        historical_features['prev_accuracy'] = [
            self._get_historical_accuracy_up_to(i) 
            for i in range(len(historical_features))
        ]
        
        return historical_features, historical_target

    def _get_historical_accuracy_up_to(self, index):
        """获取到指定索引位置的历史准确率"""
        if index < 1:
            return 0
        subset = self.history_predictions[:index]
        correct = sum(1 for p in subset if 'is_correct' in p and p['is_correct'])
        return correct / len(subset) if subset else 0

    def generate_html_report(self, filename='stock_report.html'):
        """生成并保存可视化HTML报告"""
        result = self.predict_probability()
        
        # 动态获取特征重要性/系数
        model = self.best_model.named_steps['model']
        
        # 检查模型类型
        if hasattr(model, 'feature_importances_'):
            importance_values = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 取系数的绝对值并平均多分类情况
            coefs = np.abs(model.coef_)
            if coefs.ndim > 1:  # 处理多分类情况
                importance_values = coefs.mean(axis=0)
            else:
                importance_values = coefs
        else:
            importance_values = np.zeros(len(self.selected_features))
            
        feature_importances = pd.Series(
            importance_values,
            index=self.selected_features
        ).sort_values(ascending=False)
        
        # 创建特征重要性图表
        importance_fig = go.Figure()
        importance_fig.add_trace(go.Bar(
            x=feature_importances.values,
            y=feature_importances.index,
            orientation='h',
            marker_color='#3498db'
        ))
        importance_fig.update_layout(
            title='特征重要性排名',
            height=600,
            width=800,
            showlegend=False
        )
        
        # 获取最近5个交易日的技术指标
        tech_indicators = self.data[['MACD', 'RSI', 'BB_Width', 'ATR', 'OBV']].tail(5)
        tech_table = go.Figure(data=[go.Table(
            header=dict(
                values=['日期'] + tech_indicators.columns.tolist(),
                fill_color='#3498db',
                font_color='white'
            ),
            cells=dict(
                values=[tech_indicators.index.strftime('%Y-%m-%d').tolist()] + 
                       [tech_indicators[col].round(2) for col in tech_indicators.columns],
                fill_color='#f8f9fa'
            )
        )])
        tech_table.update_layout(
            title='近期技术指标值',
            height=300,
            width=800
        )
        
        # 创建图表
        fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'xy'}]])
        
        # 饼状图
        fig.add_trace(go.Pie(
            labels=['上涨概率', '下跌概率'],
            values=[result['rise_prob'], result['fall_prob']],
            name="概率分布",
            hole=0.3,
            marker_colors=['#2ecc71', '#e74c3c']
        ), row=1, col=1)
        
        # 柱状图
        fig.add_trace(go.Bar(
            x=['上涨概率', '下跌概率'],
            y=[result['rise_prob'], result['fall_prob']],
            marker_color=['#2ecc71', '#e74c3c'],
            text=[f"{result['rise_prob']:.1f}%", f"{result['fall_prob']:.1f}%"],
            textposition='auto'
        ), row=1, col=2)
        
        fig.update_layout(
            title_text=f'{result["ticker"]} 涨跌概率分析',
            height=400,
            width=800
        )
        
        # HTML模板
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>股票预测报告 - {result["ticker"]}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                .header {{ 
                    background: linear-gradient(45deg, #3498db, #2c3e50);
                    color: white; 
                    padding: 2rem; 
                    border-radius: 10px;
                    margin-bottom: 2rem;
                }}
                .metrics {{ 
                    display: grid; 
                    grid-template-columns: repeat(2, 1fr); 
                    gap: 1rem; 
                    margin-bottom: 2rem;
                }}
                .metric-box {{
                    background: #f8f9fa;
                    padding: 1.5rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .chart-container {{ 
                    margin: 2rem 0; 
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 1rem;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{result["ticker"]} 股票预测报告</h1>
                <p>报告日期：{result["date"]}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-box">
                    <h3>最新收盘价</h3>
                    <p style="color: #27ae60; font-size: 2rem;">¥{result["close_price"]:.2f}</p>
                </div>
                <div class="metric-box">
                    <h3>预测模型</h3>
                    <p style="color: #2980b9; font-size: 1.5rem;">{result["model"]}</p>
                </div>
            </div>
            
            <div class="chart-container">
                {fig.to_html(full_html=False)}
            </div>

            <div class="chart-container">
                {importance_fig.to_html(full_html=False)}
            </div>

            <div class="chart-container">
                {tech_table.to_html(full_html=False)}
            </div>

            <div class="analysis-section">
                <h2>分析过程记录</h2>
                <div class="metric-box">
                    <h3>关键决策因素</h3>
                    <ul>
                        <li>MACD直方图趋势: {self._get_macd_trend()}</li>
                        <li>RSI超买/超卖状态: {self._get_rsi_status()}</li>
                        <li>布林带宽度变化率: {self.data['BB_Width'].pct_change()[-1]:.2%}</li>
                    </ul>
                </div>
                
            <div class="metric-box">
                <h3>历史预测准确率</h3>
                <p>最近5次预测准确率: {self._get_recent_accuracy()}%</p>
                <p>总体预测准确率: {self._get_overall_accuracy()}%</p>
                <h4>本次预测验证：</h4>
                <p>预测时间: {result['date']}</p>
                <p>预测收盘价: {result['close_price']:.2f}</p>
                <p>实际收盘价: {self.history_predictions[-1]['actual'] or '待更新'}</p>
                <p>预测结果: { 
                    '待验证' if self.history_predictions[-1]['actual'] is None else 
                    ('正确' if self.history_predictions[-1]['is_correct'] else '错误')
                }</p>
            </div>
            <div class="metric-box">
                <h3>历史记录存储位置</h3>
                <p>文件路径: {os.path.abspath(self.history_file)}</p>
                <p>记录条数: {len(self.history_predictions)}</p>
            </div>
            </div>

            <style>
                .analysis-section {{
                    margin-top: 2rem;
                    padding: 1rem;
                    background: #f8f9fa;
                    border-radius: 8px;
                }}
                .analysis-section h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 0.5rem;
                }}
                .analysis-section ul {{
                    list-style-type: none;
                    padding-left: 0;
                }}
                .analysis-section li {{
                    padding: 0.5rem 0;
                    border-bottom: 1px solid #eee;
                }}
            </style>
            
        </body>
        </html>
        """
        # 保存HTML文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 自动打开报告
        abs_path = os.path.abspath(filename)
        # 转换Windows路径为URL格式
        url_path = abs_path.replace('\\', '/')
        print(f"报告已保存至：{abs_path}")
        
        try:
            # 先检查文件是否存在
            if os.path.exists(abs_path):
                # 显式使用默认浏览器打开
                webbrowser.get().open(f'file:///{url_path}', new=2)
                print("正在尝试打开浏览器显示报告...")
            else:
                print(f"错误：报告文件 {abs_path} 未找到")
        except Exception as e:
            print(f"打开浏览器失败：{str(e)}")
            print(f"请手动打开文件：{abs_path}")
        
        return html_content

    # 使用示例
if __name__ == "__main__":
    # 输入股票代码（只需数字部分）
    predictor = StockPredictor(ticker='600418.SS', end_date=datetime.datetime.now().strftime('%Y-%m-%d'))  
    predictor.train_model()
    print(predictor.predict_probability())
    predictor.generate_html_report()  # 添加生成报告调用
