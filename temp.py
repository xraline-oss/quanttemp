import akshare as ak
import baostock as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
import time
import warnings
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# ==========================================
# 0. 环境配置
# ==========================================
warnings.filterwarnings('ignore')
os.environ['http_proxy'] = "" 
os.environ['https_proxy'] = ""

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

class TF_Macro_Alpha_Debug:
    def __init__(self, start_date='20190101'):
        self.start_date = start_date
        # 预热5年
        fetch_dt = datetime.strptime(start_date, "%Y%m%d") - timedelta(days=365*5)
        self.fetch_start_date = fetch_dt.strftime("%Y-%m-%d")
        self.fetch_start_date_ak = fetch_dt.strftime("%Y%m%d")
        
        self.macro_raw = pd.DataFrame()
        self.macro_derived = pd.DataFrame()
        self.sector_rets = pd.DataFrame()
        self.signals = pd.DataFrame()
        self.explainer = pd.Series()
        
        # 资产池
        self.sector_pool = {
            'finance': ['sh.601318', 'sh.600036', 'sh.601166', 'sh.600000', 'sh.600030', 'sh.601398', 'sh.601288', 'sh.601939', 'sh.601328', 'sh.601009'],
            'cycle_up': ['sh.601899', 'sh.600547', 'sh.600111', 'sh.600988', 'sh.600489', 'sh.603799', 'sh.600028', 'sh.601088', 'sh.601225', 'sz.002460'],
            'cycle_mid': ['sh.600309', 'sh.600019', 'sh.600585', 'sz.000778', 'sh.600010', 'sh.601600', 'sz.000060', 'sh.600426', 'sh.600104', 'sz.002493'],
            'cycle_down': ['sh.600031', 'sz.000425', 'sh.601668', 'sh.601800', 'sh.601186', 'sz.000002', 'sh.600048', 'sh.600383', 'sh.600507', 'sh.600720'],
            'stable': ['sh.600900', 'sh.600027', 'sh.600011', 'sh.600795', 'sh.600674', 'sh.600886', 'sh.600023', 'sh.601901', 'sh.600905', 'sh.601111'],
            'consumption': ['sh.600519', 'sz.000858', 'sh.600887', 'sz.002304', 'sh.600600', 'sh.600276', 'sz.000651', 'sz.000333', 'sh.603288', 'sh.600009'],
            'growth': ['sz.300750', 'sz.002594', 'sz.300015', 'sh.603501', 'sz.002475', 'sz.300124', 'sz.300274', 'sz.002466', 'sh.600438', 'sz.002236']
        }
        
        self.sector_cn = {
            'finance': '金融', 'cycle_up': '周期上游', 'cycle_mid': '周期中游',
            'cycle_down': '周期下游', 'stable': '稳定', 'consumption': '消费', 'growth': '成长'
        }

        self.regime_allocation = {
            (1, -1): ['cycle_up', 'cycle_mid', 'growth'],
            (1, 1):  ['cycle_mid', 'finance'],
            (-1, 1): ['stable', 'consumption'],
            (-1, -1):['growth', 'consumption']
        }
        self.regime_map = {
            (1, -1): "I: 复苏 (Credit↑ Rate↓)", (1, 1): "II: 过热 (Credit↑ Rate↑)",
            (-1, 1): "III: 滞胀 (Credit↓ Rate↑)", (-1, -1): "IV: 衰退 (Credit↓ Rate↓)"
        }

    # ==========================================
    # 工具函数
    # ==========================================
    def retry(self, func, *args, **kwargs):
        for i in range(3):
            try:
                res = func(*args, **kwargs)
                if res is not None and not res.empty: return res
            except: time.sleep(0.5)
        return None

    def clean_date_and_resample(self, df, col, name):
        """核心修复：强制日期清洗并对齐到月末"""
        if df is None or df.empty: return None
        try:
            # 1. 暴力清洗日期字符串
            df['idx_date'] = df[col].astype(str).str.replace(r'[年月/]', '-', regex=True).str.replace('份','').str.strip()
            df['idx_date'] = pd.to_datetime(df['idx_date'], errors='coerce')
            
            # 2. 剔除无效日期
            df = df.dropna(subset=['idx_date']).set_index('idx_date').sort_index()
            
            # 3. 寻找数值列
            val_col = None
            for c in df.columns:
                # 排除明显不是数据的列
                if '日' not in c and '月' not in c and '时间' not in c and 'date' not in c.lower():
                    # 优先找包含特定关键词的列
                    if name == 'PMI' and '制造业' in c and '非' not in c: val_col = c; break
                    if name in ['M1', 'M2'] and name in c and '同比' in c: val_col = c; break
                    if name in ['CPI', 'PPI', 'Exports'] and '同比' in c: val_col = c; break
                    
            if not val_col: 
                # 兜底：取最后一列
                val_col = df.columns[-1]
            
            # 4. 转数值并重采样到月末 (M)
            # 这一步至关重要，它保证了所有数据都在同一个时间轴上
            series = pd.to_numeric(df[val_col], errors='coerce').resample('M').last()
            
            # 打印调试信息
            # print(f"   [调试] {name}: 获取{len(series)}行, 范围 {series.index[0].date()}~{series.index[-1].date()}")
            return series
            
        except Exception as e:
            print(f"   [调试] {name} 解析失败: {e}")
            return None

    # ==========================================
    # 1. 数据获取 (逐个击破)
    # ==========================================
    def fetch_data(self):
        print(f"1. 启动数据获取 (数据源: Akshare & Baostock)...")
        macro_dict = {}
        
        # --- A. 逐个获取宏观数据 ---
        
        # 1. PMI
        df = self.retry(ak.macro_china_pmi)
        macro_dict['PMI'] = self.clean_date_and_resample(df, '月份', 'PMI')

        # 2. 货币供应 (M1 & M2)
        # 注意：M1和M2在同一个接口，这里调用两次分别提取
        df_money = self.retry(ak.macro_china_money_supply)
        if df_money is not None:
            # 专门提取M2
            m2_col = [c for c in df_money.columns if 'M2' in c and '同比' in c]
            if m2_col:
                temp = df_money[['月份', m2_col[0]]].copy()
                macro_dict['M2'] = self.clean_date_and_resample(temp, '月份', 'M2')
            
            # 专门提取M1
            m1_col = [c for c in df_money.columns if 'M1' in c and '同比' in c]
            if m1_col:
                temp = df_money[['月份', m1_col[0]]].copy()
                macro_dict['M1'] = self.clean_date_and_resample(temp, '月份', 'M1')

        # 3. CPI
        df = self.retry(ak.macro_china_cpi)
        macro_dict['CPI'] = self.clean_date_and_resample(df, '月份', 'CPI')

        # 4. PPI
        df = self.retry(ak.macro_china_ppi)
        macro_dict['PPI'] = self.clean_date_and_resample(df, '月份', 'PPI')
        
        # 5. 利率 (国债) - 最容易失败的
        s_bond = None
        try:
            df = ak.bond_zh_us_rate(start_date="20100101")
            s_bond = self.clean_date_and_resample(df, '日期', 'Bond_10y')
        except: pass
        
        # 如果国债失败，尝试备用接口 (bond_china_yield)
        if s_bond is None:
            try:
                df = ak.bond_china_yield(start_date="20100101", end_date=datetime.now().strftime("%Y%m%d"))
                df = df[df['曲线名称']=='中债国债收益率曲线']
                # 手动指定列名提取
                temp = df[['日期', '10年']].copy()
                s_bond = self.clean_date_and_resample(temp, '日期', 'Bond_10y')
            except: pass
            
        macro_dict['Bond_10y'] = s_bond

        # --- 数据完整性检查 ---
        valid_count = sum(1 for v in macro_dict.values() if v is not None)
        print(f"   -> 成功获取宏观指标数: {valid_count}/6")
        
        if valid_count < 3:
            print("❌ 错误：宏观数据严重缺失，无法构建因子。")
            return False

        # 合并宏观数据 (使用 Outer Join 保留所有数据，然后前值填充)
        self.macro_raw = pd.concat(macro_dict, axis=1).sort_index().ffill()
        
        # 再次检查：必须要有数据
        if self.macro_raw.dropna(how='all').empty:
            print("❌ 错误：合并后宏观数据为空。")
            return False

        # --- B. 板块行情 (Baostock) ---
        bs.login()
        sector_price_dict = {}
        for sector, codes in self.sector_pool.items():
            pool_df = pd.DataFrame()
            for code in codes:
                try:
                    rs = bs.query_history_k_data_plus(code, "date,close", 
                        start_date=self.fetch_start_date, end_date=datetime.now().strftime("%Y-%m-%d"), 
                        frequency="d", adjustflag="3")
                    if rs.data:
                        data = pd.DataFrame(rs.data, columns=rs.fields)
                        data['date'] = pd.to_datetime(data['date'])
                        pool_df[code] = data.set_index('date')['close'].astype(float)
                except: pass
            if not pool_df.empty:
                sector_price_dict[sector] = pool_df.mean(axis=1).resample('M').last()
        bs.logout()
        
        self.sector_rets = pd.DataFrame(sector_price_dict).pct_change().dropna()
        
        # --- C. 最终对齐 ---
        common = self.macro_raw.index.intersection(self.sector_rets.index)
        self.macro_raw = self.macro_raw.loc[common]
        self.sector_rets = self.sector_rets.loc[common]
        
        if len(self.macro_raw) < 24: # 至少需要24个月数据
            print(f"❌ 错误：有效重叠数据不足 (仅 {len(self.macro_raw)} 个月)。")
            return False
            
        return True

    # ==========================================
    # 2. 经济学特征工程
    # ==========================================
    def engineer_factors(self):
        print("2. 构建经济学衍生因子...")
        df = self.macro_raw.copy()
        
        # 容错计算：如果缺列，就跳过该因子
        if 'M2' in df and 'PPI' in df:
            df['Excess_Liquidity'] = df['M2'] - df['PPI'] # 剩余流动性
        else: df['Excess_Liquidity'] = 0
            
        if 'PPI' in df and 'CPI' in df:
            df['Profit_Scissors'] = df['PPI'] - df['CPI'] # 剪刀差
        else: df['Profit_Scissors'] = 0
            
        if 'Bond_10y' in df and 'CPI' in df:
            df['Real_Rate'] = df['Bond_10y'] - df['CPI'] # 实际利率
        else: df['Real_Rate'] = 0
            
        if 'M1' in df and 'M2' in df:
            df['M1_M2_Gap'] = df['M1'] - df['M2'] # 活化度
        else: df['M1_M2_Gap'] = 0
            
        # 填充缺失计算
        self.macro_derived = df.ffill().fillna(0)
        
        # 滚动分位数 (Rolling Percentile)
        self.macro_rank = self.macro_derived.rolling(window=36, min_periods=12).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        self.macro_rank = self.macro_rank.dropna()

    # ==========================================
    # 3. 机器学习
    # ==========================================
    def train_constrained_model(self):
        print("3. 运行 LightGBM (带约束)...")
        
        # 动态筛选存在的列
        avail_cols = self.macro_rank.columns.tolist()
        
        cf_feats = [f for f in ['PMI', 'Excess_Liquidity', 'M1_M2_Gap'] if f in avail_cols]
        dr_feats = [f for f in ['Real_Rate', 'Bond_10y', 'Profit_Scissors'] if f in avail_cols]
        
        # 确保至少有一个因子
        if not cf_feats: cf_feats = ['M2'] if 'M2' in avail_cols else []
        if not dr_feats: dr_feats = ['CPI'] if 'CPI' in avail_cols else []
        
        used_feats = cf_feats + dr_feats
        
        # 对齐
        common_idx = self.macro_rank.index.intersection(self.sector_rets.index)
        X = self.macro_rank.loc[common_idx, used_feats]
        y = self.sector_rets.loc[common_idx].mean(axis=1)
        
        # 经济学约束
        mc = [1 if f in cf_feats else -1 for f in used_feats]
            
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, 
                                  monotone_constraints=mc, random_state=42, verbose=-1)
        model.fit(X, y)
        
        self.explainer = pd.Series(model.feature_importances_, index=X.columns)
        if self.explainer.sum() > 0: self.explainer /= self.explainer.sum()
        
        # 合成
        def calc_score(feats):
            if not feats: return 0
            w = self.explainer[feats]
            if w.sum() > 0: w /= w.sum()
            return (self.macro_rank[feats] * w).sum(axis=1)
            
        self.signals['Syn_CF'] = calc_score(cf_feats)
        self.signals['Syn_DR'] = calc_score(dr_feats)

    # ==========================================
    # 4. 评估与决策
    # ==========================================
    def evaluate_and_decide(self):
        if self.signals.empty: return
        df = self.signals.copy()
        
        df['Signal_CF'] = np.where(df['Syn_CF'] > 0.5, 1, -1)
        df['Signal_DR'] = np.where(df['Syn_DR'] > 0.5, 1, -1)
        
        # AUC & Win Rate
        next_ret = self.sector_rets.shift(-1).loc[df.index]
        valid_mask = next_ret.notna().all(axis=1)
        
        auc_score, win_rate = 0.5, 0.5
        if valid_mask.sum() > 10:
            y_true = (next_ret['cycle_up'] > next_ret['stable']).astype(int)
            try: auc_score = roc_auc_score(y_true[valid_mask], df.loc[valid_mask, 'Syn_CF'])
            except: pass
            
            hits = []
            for idx, row in df[valid_mask].iterrows():
                cf, dr = row['Signal_CF'], row['Signal_DR']
                targets = self.regime_allocation.get((cf, dr), [])
                valid_t = [t for t in targets if t in next_ret.columns]
                if valid_t:
                    hits.append(1 if next_ret.loc[idx, valid_t].mean() > next_ret.loc[idx].mean() else 0)
            if hits: win_rate = np.mean(hits)

        # Output
        latest = df.iloc[-1]
        latest_raw = self.macro_derived.iloc[-1]
        
        cf_trend = 1 if latest['Syn_CF'] > 0.5 else -1
        dr_trend = 1 if latest['Syn_DR'] > 0.5 else -1
        regime = self.regime_map.get((cf_trend, dr_trend))
        targets = self.regime_allocation.get((cf_trend, dr_trend))
        target_cn = [self.sector_cn[t] for t in targets]
        
        print("\n" + "="*60)
        print("🚀 [Macro-Hedge Alpha] 宏观对冲决策报告")
        print("="*60)
        print(f"数据截止: {latest.name.strftime('%Y-%m-%d')}")
        print(f"模型回测: 信号AUC={auc_score:.2f} | 历史胜率={win_rate:.1%}")
        print("-" * 60)
        
        print("【因子归因】")
        cf_status = "📈 扩张" if cf_trend==1 else "📉 收缩"
        print(f"1. 现金流 (CF) -> {cf_status} (得分: {latest['Syn_CF']:.0%})")
        
        dr_status = "📈 收紧" if dr_trend==1 else "📉 宽松"
        print(f"2. 折现率 (DR) -> {dr_status} (得分: {latest['Syn_DR']:.0%})")
            
        print("-" * 60)
        print(f"【最终判决】")
        print(f"  当前处于:  {regime}")
        print(f"  建议超配:  {target_cn}")
        print("="*60)

    def run(self):
        if self.fetch_data():
            self.engineer_factors()
            self.train_constrained_model()
            self.evaluate_and_decide()
        else:
            print("数据获取失败，请检查网络。")

if __name__ == '__main__':
    model = TF_Macro_Alpha_Debug(start_date='20190101')
    model.run()