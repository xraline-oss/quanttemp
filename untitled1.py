import akshare as ak
import baostock as bs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import warnings
from datetime import datetime, timedelta

# ==========================================
# 0. 环境配置
# ==========================================
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

class TF_Macro_Alpha_Final:
    def __init__(self, start_date='20190101'):
        self.start_date = start_date
        fetch_dt = datetime.strptime(start_date, "%Y%m%d") - timedelta(days=365*5)
        self.fetch_start_date = fetch_dt.strftime("%Y-%m-%d")
        
        self.macro_raw = pd.DataFrame()
        self.sector_rets = pd.DataFrame()
        self.signals = pd.DataFrame()
        
        # 资产池
        self.sector_pool = {
            'finance': ['sh.601318', 'sh.600036', 'sh.601166', 'sh.600000', 'sh.600030'],
            'cycle_up': ['sh.601899', 'sh.600547', 'sh.600111', 'sh.600988', 'sh.600489'],
            'cycle_mid': ['sh.600309', 'sh.600019', 'sh.600585', 'sz.000778', 'sh.600010'],
            'cycle_down': ['sh.600031', 'sz.000425', 'sh.601668', 'sh.601800', 'sh.601186'],
            'stable': ['sh.600900', 'sh.600027', 'sh.600011', 'sh.600795', 'sh.600674'],
            'consumption': ['sh.600519', 'sz.000858', 'sh.600887', 'sz.002304', 'sh.600600'],
            'growth': ['sz.300750', 'sz.300015', 'sz.300015', 'sh.603501', 'sz.002475']
        }
        
        self.sector_cn = {
            'finance': '金融', 'cycle_up': '周期上游', 'cycle_mid': '周期中游',
            'cycle_down': '周期下游', 'stable': '稳定', 'consumption': '消费', 'growth': '成长'
        }
        
        self.allocation_map = {
            (1, -1): ['finance', 'cycle_up', 'cycle_mid'],
            (1, 1):  ['finance', 'consumption'],
            (-1, 1): ['consumption', 'stable', 'cycle_down'],
            (-1, -1):['growth', 'consumption']
        }

    def retry(self, func, *args, **kwargs):
        for i in range(3):
            try:
                res = func(*args, **kwargs)
                if res is not None and not res.empty: return res
            except: time.sleep(1)
        return None

    def clean_date_and_resample(self, df, name, date_col=None, val_col_name=None, keywords=None):
        if df is None or df.empty: return None
        try:
            temp = df.copy()
            # 1. 自动定位日期列
            if date_col is None:
                for c in temp.columns:
                    if '日期' in c or '月份' in c or 'date' in c.lower():
                        date_col = c
                        break
            if date_col is None or date_col not in temp.columns:
                return None
                
            # 2. 自动定位数值列 (增强版逻辑)
            target_col = None
            if val_col_name and val_col_name in temp.columns:
                target_col = val_col_name
            elif keywords:
                # 只要包含任意一个核心关键词即可，放宽条件
                for c in temp.columns:
                    if c == date_col: continue
                    # 修改：all -> any (或者只取第一个匹配的)
                    # 这里逻辑：必须包含所有keywords中的词
                    if all(k in c for k in keywords):
                        target_col = c
                        break
            
            # 兜底：如果没找到，且只有2列，取非日期列
            if target_col is None and len(temp.columns) == 2:
                target_col = [c for c in temp.columns if c != date_col][0]

            if target_col is None:
                return None
            
            # 3. 清洗日期
            temp['idx_date'] = temp[date_col].astype(str).str.replace(r'[年月/]', '-', regex=True).str.replace('份','').str.strip()
            temp['idx_date'] = pd.to_datetime(temp['idx_date'], errors='coerce')
            temp = temp.dropna(subset=['idx_date']).set_index('idx_date').sort_index()
            
            # 4. 转数值
            series = pd.to_numeric(temp[target_col], errors='coerce').resample('M').last()
            return series
            
        except Exception:
            return None

    # ==========================================
    # 1. 数据获取
    # ==========================================
    def fetch_data(self):
        print("1. 启动数据获取...")
        macro_dict = {}
        
        # --- 改进关键词，提高成功率 ---
        print("   > 获取 PMI...")
        df = self.retry(ak.macro_china_pmi)
        # 修改：关键词只用 ['PMI']，不再强制要求 '制造业'
        macro_dict['PMI'] = self.clean_date_and_resample(df, 'PMI', keywords=['PMI'])

        print("   > 获取 工业增加值...")
        df = self.retry(ak.macro_china_industrial_production_yoy)
        # 修改：关键词只用 ['同比']
        macro_dict['Ind_Val'] = self.clean_date_and_resample(df, 'Ind_Val', keywords=['同比'])

        print("   > 获取 国债收益率...")
        df_bond = self.retry(ak.bond_zh_us_rate)
        if df_bond is not None:
            s_10y = self.clean_date_and_resample(df_bond, 'Bond_10y', val_col_name='中国国债收益率10年')
            if s_10y is None: s_10y = self.clean_date_and_resample(df_bond, 'Bond_10y', keywords=['中国', '10年'])
            macro_dict['Bond_10y'] = s_10y
            
            s_2y = self.clean_date_and_resample(df_bond, 'Bond_2y', val_col_name='中国国债收益率2年')
            if s_2y is None: s_2y = self.clean_date_and_resample(df_bond, 'Bond_2y', keywords=['中国', '2年'])
            
            if s_10y is not None and s_2y is not None:
                common = s_10y.index.intersection(s_2y.index)
                macro_dict['Term_Spread'] = s_10y.loc[common] - s_2y.loc[common]

        print("   > 获取 货币供应量 (M1/M2)...")
        df_money = self.retry(ak.macro_china_money_supply)
        if df_money is not None:
            macro_dict['M2'] = self.clean_date_and_resample(df_money, 'M2', keywords=['M2', '同比'])
            macro_dict['M1'] = self.clean_date_and_resample(df_money, 'M1', keywords=['M1', '同比'])

        valid_keys = [k for k,v in macro_dict.items() if v is not None]
        print(f"   -> 成功获取: {valid_keys}")
        
        self.macro_raw = pd.concat(macro_dict, axis=1).sort_index().ffill()

        print("   > 获取 板块行情 (Baostock)...")
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
        
        common = self.macro_raw.index.intersection(self.sector_rets.index)
        if len(common) == 0:
            print("❌ 宏观数据与行情数据无交集，无法回测")
            return False
            
        self.macro_raw = self.macro_raw.loc[common]
        self.sector_rets = self.sector_rets.loc[common]
        
        print(f"   数据准备完成，有效样本数: {len(common)}")
        return True

    # ==========================================
    # 2. 策略逻辑 (修复 Crash 部分)
    # ==========================================
    def run_strategy(self):
        print("2. 运行策略模型 (文献复现)...")
        try:
            df = self.macro_raw.copy()
            df = df.shift(1).dropna()
            
            ma6 = df.rolling(6).mean()
            trend = (df > ma6).astype(int).replace(0, -1)
            
            # --- 核心修复: 确保是 Series 格式 ---
            # 如果缺少某列，trend['col'] 会报错，或者如果用 get 可能会返回 None/int
            # 我们强制构造一个全0的 Series 作为 fallback
            zeros = pd.Series(0, index=df.index)
            
            # 使用 ternary operator 确保结果一定是 Series
            pmi_trend = trend['PMI'] if 'PMI' in trend else zeros
            ind_trend = trend['Ind_Val'] if 'Ind_Val' in trend else zeros
            bond_trend = trend['Bond_10y'] if 'Bond_10y' in trend else zeros
            
            # 合成信号 (Series + Series = Series)
            cf_score = pmi_trend + ind_trend
            
            dr_score = bond_trend
            if 'Term_Spread' in trend:
                dr_score += trend['Term_Spread'] 

            self.signals = pd.DataFrame(index=df.index)
            # 此时 cf_score 必然是 Series，可以安全调用 apply
            self.signals['CF_Dir'] = cf_score.apply(lambda x: 1 if x > 0 else -1)
            self.signals['DR_Dir'] = dr_score.apply(lambda x: 1 if x > 0 else -1)
            
            print(f"   策略计算完成，生成信号 {len(self.signals)} 期")
            return True
            
        except Exception as e:
            print(f"❌ 策略计算出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def report(self):
        if self.signals.empty: return
        
        last_date = self.signals.index[-1]
        last_sig = self.signals.iloc[-1]
        cf, dr = last_sig['CF_Dir'], last_sig['DR_Dir']
        
        print("\n" + "="*60)
        print(f"🚀 [天风研报复现] 宏观对冲策略报告")
        print(f"📅 最新信号日期: {last_date.strftime('%Y-%m-%d')}")
        print(f"   现金流 (CF): {'📈 扩张' if cf==1 else '📉 收缩'}")
        print(f"   折现率 (DR): {'📈 收紧' if dr==1 else '📉 宽松'}")
        
        target_sectors = self.allocation_map.get((cf, dr), [])
        target_cn = [self.sector_cn.get(s, s) for s in target_sectors]
        print(f"   -> 建议配置: {target_cn}")
        print("="*60)

        self.backtest()

    def backtest(self):
        print("3. 执行回测并绘图...")
        try:
            common_idx = self.signals.index.intersection(self.sector_rets.index)
            if len(common_idx) < 10:
                print("样本太少，跳过回测")
                return
                
            sig_slice = self.signals.loc[common_idx]
            ret_slice = self.sector_rets.shift(-1).loc[common_idx].fillna(0)
            
            strategy_ret = []
            for date, row in sig_slice.iterrows():
                cf, dr = row['CF_Dir'], row['DR_Dir']
                targets = self.allocation_map.get((cf, dr), [])
                valid = [t for t in targets if t in ret_slice.columns]
                r = ret_slice.loc[date, valid].mean() if valid else 0
                strategy_ret.append(r)
            
            s_cum = (1 + pd.Series(strategy_ret, index=common_idx)).cumprod()
            b_cum = (1 + ret_slice.mean(axis=1)).cumprod()
            
            print(f"   策略累计净值: {s_cum.iloc[-1]:.4f}")
            print(f"   基准累计净值: {b_cum.iloc[-1]:.4f}")
            
            plt.figure(figsize=(12, 6))
            b_cum.plot(label='Benchmark', color='gray', linestyle='--')
            s_cum.plot(label='Macro Strategy', color='red', linewidth=2)
            plt.title('Replication: TianFeng Macro Sector Rotation')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except Exception as e:
            print(f"❌ 回测出错: {e}")

if __name__ == '__main__':
    app = TF_Macro_Alpha_Final(start_date='20190101')
    if app.fetch_data():
        if app.run_strategy():
            app.report()