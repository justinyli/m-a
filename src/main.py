import pandas as pd

merger_path = 'src\\data\\mergers.xlsx'

mergers = pd.DataFrame()
with pd.ExcelFile(merger_path) as xls:
    for sheet_name in xls.sheet_names:
        sheet_df = xls.parse(sheet_name)
        mergers = pd.concat([mergers, sheet_df], ignore_index=True)

nonmerger_path = 'src\\data\\nonmergers.xlsx'

nonmergers = pd.DataFrame()
with pd.ExcelFile(nonmerger_path) as xls:
    for sheet_name in xls.sheet_names:
        sheet_df2 = xls.parse(sheet_name)
        nonmergers = pd.concat([nonmergers, sheet_df2], ignore_index=True)

data = pd.concat([mergers, nonmergers], ignore_index=True)


data = data.drop(columns=[
    'AD-3',
    'AD-2',
    'AD-1',
    'AD-30',
    'Announcement Date',
    'Company RIC'
])
data = data.dropna()

print(data)