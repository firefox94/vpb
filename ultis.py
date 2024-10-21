import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import unidecode

def miss_report(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Create a missing values report table for each of columns in df \n
    Return: DataFrame
    """
    dic = {}
    for col in df.columns:
        miss = df[col].isnull().sum() / df.shape[0]
        dic[col] = miss
        miss_report = pd.DataFrame(list(dic.items()), columns=['Column','MissingRatio'])
    miss_report['MissingRatio'] = miss_report['MissingRatio'].astype(float)
    # miss_report['MissingRatio'] = miss_report['MissingRatio'].apply(lambda x: "{:.4}".format(x))
    miss_report.sort_values(by='MissingRatio', ascending=False)
    return miss_report


def generate_miss_report(data: pd.DataFrame) -> pd.DataFrame:
    """ 
    Create a missing values report table for each of columns in df \n
    Return: DataFrame
    """
    # Define missing/invalid patterns 
    ch_miss = ['n/a','na','na.','n.a.','n.a','*','-','unknown','email@domain.com',
               'testuser','u','99999999','null','none','c9999','z_error','z_missing','',' ','unspecified','nan']
    nm_miss = [99999999.0,-999.0]
    
    def set_nan(x):
        if type(x) is str :
            if x.lower() in ch_miss :
                return np.nan
            else:
                return x
        elif type(x) is int or type(x) is float:
            if x in nm_miss:
                return np.nan
            else:
                return x
        else:
             return x

    data_dub = data.applymap(set_nan)
    n_miss  = data_dub.isnull().sum() 
    p_miss  = round(data_dub.isnull().sum() / len(data_dub.index),2)
    n_unique = data_dub.nunique()
    p_unique = round(n_unique / len(data_dub.index),2)
    miss_report = pd.merge(n_miss.rename('n_miss'),p_miss.rename('%miss'),left_index=True,right_index=True)
    miss_report = miss_report.merge(n_unique.rename('n_unique'),left_index=True,right_index=True)
    miss_report = miss_report.merge(p_unique.rename('%unique'),left_index=True,right_index=True)
    miss_report['n_populated'] = len(data_dub.index) - miss_report['n_miss']
    miss_report['%populated'] = round(1.00 - miss_report['%miss'],2)
    miss_report.reset_index(inplace=True)
    miss_report.rename(columns={'index':'features'},inplace=True)
    return miss_report


def visual_distribution(df: pd.DataFrame,col: str,label: int|bool,sortby: str = 'percent') -> plt.Axes: 
    """
    Create a visualize distribution of LABEL in a column \n
    Return: Axes
    """
    pivot = df.pivot_table(index=col, columns=label, aggfunc='size').reset_index()
    pivot.columns = [col,'NO','YES']
    pivot['total'] = pivot['YES'] + pivot['NO']
    pivot['percent'] = pivot['YES'] / pivot['NO']
    print(pivot.sort_values(by=sortby, ascending=False))
    return pivot.plot(x=col, y=['NO','YES'], kind='bar')


def create_gainlift_table(data,var_to_rank,var_to_count_nonzero,n_qcut=10):
    tmpname = var_to_rank+'_tmp'
    data[tmpname] = pd.qcut(data[var_to_rank].rank(method='first'), n_qcut, labels=False)
    
    table = pd.pivot_table(data, 
                           index=[tmpname],
                           values=[var_to_rank,var_to_count_nonzero],
                           aggfunc={
                               var_to_rank:np.size,
                               var_to_count_nonzero:np.count_nonzero
                           }
                          )
    table_sorted = table.sort_index(ascending=False)
    table_sorted['cumulative_response'] = table_sorted[var_to_count_nonzero].cumsum()
    
    table_sorted['nonresponse'] = table_sorted[var_to_rank] - table_sorted[var_to_count_nonzero]
    table_sorted['cumulative_nonresponse'] = table_sorted['nonresponse'].cumsum()
    
    total_nonresponse = np.sum(table_sorted.loc[:,'nonresponse'])
    table_sorted['percent_of_nonevents'] = (table_sorted['nonresponse']/total_nonresponse)
    table_sorted['cumulative_percent_of_nonevents'] = table_sorted['percent_of_nonevents'].cumsum()
    
    total_response = np.sum(table_sorted.loc[:,var_to_count_nonzero])
    table_sorted['percent_of_events'] = (table_sorted[var_to_count_nonzero]/total_response)
    table_sorted['cumulative_percent_of_events'] = table_sorted['percent_of_events'].cumsum()
    
    table_sorted['cumulative_gain'] = (table_sorted['cumulative_response']/total_response)*100
    decile = np.linspace(1,n_qcut,n_qcut)
    table_sorted['decile'] = decile*(100/n_qcut)
    table_sorted['cumulative_lift'] = table_sorted['cumulative_gain']/(table_sorted['decile'])
    table_sorted.rename(columns={var_to_rank: 'counts'},inplace=True)
    table_sorted['ks'] = np.abs(table_sorted['cumulative_percent_of_events'] - table_sorted['cumulative_percent_of_nonevents'])
    table_sorted.index.names = ['rank']
    table_sorted['base_lift']=1
    table_sorted['base_gain']=[x for x in np.linspace(10,100,n_qcut)]
    
    data.drop(columns=[tmpname],inplace=True)
    
    return table_sorted

def convert_col_to_list(df: pd.DataFrame, col: str) -> str:
    """
    Convert a column into a () type which can be used in SQL query IN
    """
    text = df[col].unique().tolist()
    text = [f"'{i}'" for i in text]
    text = ",".join(text)
    text = f"({text})"
    return text


def create_barplot(df: pd.DataFrame, x: str, y: str) -> plt.Axes:
    """
    Create a count barplot with values at the end of bar
    """
    ax = sns.barplot(df,x=x,y=y)
    plt.title('Intensity by CIF')
    plt.tight_layout()
    for p in ax.patches:
        width = p.get_width()  # Lấy chiều rộng của từng cột
        ax.text(width + 100,  # Vị trí x của giá trị (cộng thêm khoảng cách)
                p.get_y() + p.get_height() / 2,  # Vị trí y của giá trị (giữa cột)
                f'{int(width)}',  # Giá trị để hiển thị
                ha='left',  # Căn chỉnh giá trị
                va='center')  # Căn chỉnh giá trị theo chiều dọc
    plt.show()
    

def remove_diacritics(text):
    if text is None:
        return ''  # Trả về chuỗi rỗng nếu giá trị là None
    return unidecode(text)