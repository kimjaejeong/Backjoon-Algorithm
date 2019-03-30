# SK C&C - 데이터 분석 역량 테스트 정리

- 시리즈 메서드 외우기 - 61쪽



- tips['day'].value_counts()
  - Sat / Sun / Thur / Fri 마다 몇개씩 있는지 알 수 있음.
  - value_counts 메서드는 지정한 열의 빈도를 구하는 메서드
  - value_counts(dropna=False)에서 dropna는 na를 뺄거니? False이므로 na도 포함.
- 시리즈 인덱스에 관계없이 막 섞기.

```python
import random

random.seed(40)
random.shuffle(scientists['Age'])

```

- 데이터프레임 열 삭제

```python
scientists_dropped = scientists.drop(['Age'], axis=1)
#axis = 1을 반드시 해줘야 함.
```

- 파일 저장 방법

```python
save_scientists 데이터 프레임이 주어졌을 때

save_scientists.to_csv('C:/doit_pandas-master/output/save_scientists.csv(원하는 파일명)')


```

- matplotlib 라이브러리로 그래프 그리기

```python
#각 subplot에 넣을 데이터 프레임 저장.
dataset_1
dataset_2
dataset_3
dataset_4

#1.전체 그래프가 위치할 기본 틀을 만든다.
fig = plt.figure()

#2.그래프를 그려 넣을 그래프 격자를 만든다.
axes1 = fig.add_subplot(2,2,1)
axes2 = fig.add_subplot(2,2,2)
axes3 = fig.add_subplot(2,2,3)
axes4 = fig.add_subplot(2,2,4)

#3. 격자에 그래프를 하나씩 추가한다. 격자에 그래프가 추가되는 순서는 왼쪽에서 오른쪽 방향.
axes1.plot(dataset_1['x'], dataset_1['y'], 'o')
axes2.plot(dataset_2['x'], dataset_2['y'])
axes3.plot(dataset_3['x'], dataset_3['y'])
axes4.plot(dataset_4['x'], dataset_4['y'])

#4-1.각 그래프에 해당하는 title 넣기.
axes1.set_title("dataset_1")
axes2.set_title("dataset_2")
axes3.set_title("dataset_3")
axes4.set_title("dataset_4")

#4-2. 전체 틀에 해당하는 제목 넣기.
fig.suptitle("Anscombe Data")

#4.그래프가 겹쳐보일 때 넣기
fig.tight_layout()
```



```python

```

- 그래프 종류

  - hist

    ```python
    import matplotlib.pyplot as plt
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    
    ax1.hist(tips['total_bill'])
    ax1.set_title('Histogram of Total Bill')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Total Bill')
    ```

    

  - scatter

    ```python
    scatter_plot = plt.figure()
    ax1 = scatter_plot.add_subplot(1,1,1)
    ax1.scatter(tips['total_bill'], tips['tip'])
    ax1.set_title('Scatterplot of Total Bill vs Tip')
    ax1.set_xlabel('Total Bill')
    ax1.set_ylabel('Tip')
    ```

  - boxplot

    ```python
    boxplot = plt.figure()
    
    ax1 = boxplot.add_subplot(1,1,1)
    ax1.boxplot([Female_box, Male_box], labels = ['Female', 'Male'])
    ax1.set_title('Boxplot of Tips by Sex')
    ax1.set_xlabel('Sex')
    ax1.set_ylabel('Tip')
    
    ```

  - 다변량 그래프 그리기

    ```python
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.scatter(x=tips['total_bill'], y=tips['tip'], c = tips['num_sex'], s = tips['size']*10, alpha=0.5)
    #c => color(점의 색상) / s => size(점의 크기) / alpha => 점의 투명도
    ax1.set_title('scatter_plot total_bill & tip')
    ax1.set_xlabel('total_bill')
    ax1.set_ylabel('tip')
    
    def record_sex(sex):
        if sex == 'Female':
            return 0
        else:
            return 1
    
    tips['num_sex'] = tips['sex'].apply(record_sex) #apply를 활용하여 Female, Male을 숫자화 시킴.
    ```

    

- seaborn 라이브러리로 그래프 그리기

  - 히스토그램 - distplot

    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    tips = sns.load_dataset('tips', engine='python')
    
    ax1 = plt.subplots()
    ax1 = sns.distplot(tips['total_bill'], kde = False)
    #kde는 밀집을 나타냄. hist=False로 두면 히스토그램이 없어짐.
    #rug=True로 두면 양탄자 그래프를 그릴 수 있음.
    ax1.set_title('Totall Bill Histogram with Density Plot')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Total Bill')
    ```

  - count 그래프

    ```python
    ax = plt.subplots()
    #ax = sns.countplot(tips['day'])
    sns.countplot(tips['day'], hue=tips['sex'], order = ['Fri', 'Sat'])
    #hue는 Female, Male에 따라 막대 2종류를 생성함.
    #order은 Fri, Sat 두개가 출력됨.
    ax.set_title('Count of days')
    ax.set_xlabel('Day of the week')
    ax.set_ylabel('Frequency')
    ```

    

  - 산점도 - regplot => hue='sex' 가 불가능

    ```python
    ax = plt.subplots()
    ax = sns.regplot(tips['total_bill'], tips['tip'])
    #fig_reg = False를 하면 회귀선이 그려지지 않음.
    #scatter = False를 하면 산점도가 그려지지 않음.
    ax.set_title('Scatter of Total Bill and Tip')
    ax.set_xlabel('Total Bill')
    ax.set_ylabel('Tip')
    ```

  - 산점도2 - lmplot => hue='sex'가 가능.

    ```python
    #예시1번
    sns.lmplot(x='total_bill', y='tip', data=tips, fit_reg=False, hue='sex', scatter_kws={'s' : tips['size']*10}, markers=['o','x'])
    #plt.subplots()를 만들 필요가 없다.
    #lmplot에서 data를 반드시 설정해줘야 함.
    #scatter_kws={'s' : tips['size']*10}은 점의 크기를 표시해준다.
    #markers는 o와 x로 표시해준다.
    
    #예시2번
    anscombe_plot = sns.lmplot(x='x', y='y', data=anscombe, fit_reg=False, col='dataset', col_wrap = 2)
    #col='dataset'을 하면 dataset을 기준으로 그래프가 나뉘어서 그려짐.
    #col_wrap을 2로 설정하면 column크기가 2 따라서 2x2가 그려짐.
    ```

    

  ​	

  - 이변량 그래프 그리기 - jointplot => 산점도 + 히스토그램

    ```python
    ax = sns.jointplot(tips['total_bill'], tips['tip'], color='blue', height=6, ratio=5, space=0.2, kind='hex')
    #kind도 여러가지 종류가 있음
    ax.set_axis_labels(xlabel = 'Total_Bill', ylabel = 'Tip')
    #jointplot일 때에는 set_axis_labels를 사용한다.
    ax.fig.suptitle('Joint Plot of Total Bill and Tip', fontsize=10, y=1.03)
    ```

  - 이변량 그래프 그리기 - kdeplot => 이차원 밀집도

    ```python
    #이차원 밀집도 그리기
    ax = plt.subplots()
    ax = sns.kdeplot(tips['total_bill'], tips['tip'], shade = True, legend= True)
    ax.set_title('Kernel Density Plot of Total Bill and Tip')
    ax.set_xlabel('Total Bill')
    ax.set_ylabel('Tip')
    ```

    

  - 바그래프 그리기

    ```python
    #시간에 따라 지불한 비용의 평균을 바 그래프로 나타내기.
    ax = plt.subplots()
    ax = sns.barplot(tips['time'], tips['total_bill'])
    ax.set_title('Bar plot of average total bill for time of day')
    ax.set_xlabel('Time of day')
    ax.set_ylabel('Average total Bill')
    ```

  - 박스 그래프 그리기

    ```python
    #박스 그래프 - 최솟값, 1분위수, 중간값, 3분위수, 최댓값, 이상치 등을 표현
    ax = plt.subplots()
    ax = sns.boxplot(tips['time'], tips['total_bill'])
    ax.set_title('Boxplot of total bill by time of day')
    ax.set_xlabel('time of day')
    ax.set_ylabel('Total Bill')
    ```

  - 바이올린 그래프 - 박스그래프 + 분산 표현 가능.

    ```python
    #바이올린 그래프 
    ax = plt.subplots()
    ax = sns.violinplot(tips['time'], tips['total_bill'])
    ax.set_title('Violin plot of total_bill by time of day')
    ax.set_xlabel('Time of day')
    ax.set_ylabel('Total Bill')
    ```

    

  - 관계 그래프

    ```python
    pair_grid = sns.PairGrid(tips) #틀 잡기.
    pair_grid = pair_grid.map_upper(sns.regplot) #대각선을 기준으로 위쪽
    pair_grid = pair_grid.map_diag(sns.distplot, rug= True)  #대각선
    pair_grid = pair_grid.map_lower(sns.kdeplot) #대각선을 기준으로 아래쪽
    ```

    

  - seaborn 라이브러리로 그래프 스타일 설정하기

    ```python
    fig = plt.figure()
    
    seaborn_styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
    
    for idx, style in enumerate(seaborn_styles):
        plot_position = idx+1
        with sns.axes_style(style):
            ax = fig.add_subplot(2,3,plot_position)
            violin = sns.violinplot(x='time', y='total_bill', data=tips)
            violin.set_title(style)
    fig.tight_layout()
    ```

    

- 데이터 연결하기

  - pd.concat 

    - 한 번에 2개 이상의 데이터 프레임을 연결할 수 있는 메서드

      ```python
      pd.concat([df1,df3], axis=1, join='inner')
      #axis=1은 columns으로 합쳐짐.
      #join='inner'는 조건에 맞는 행을 연결하는 것.
      ```

      

  - df1.append(new_row_df)

    - 연결할 데이터프레임이 1개라면 append 메서드를 사용해도 됨.

      ```python
      
      data_dict = {'A' : 'n5', 'B' : 'n6', 'C' : 'n7', 'D' : 'n8'}
      df1.append(data_dict, ignore_index=True)
      #ignore_index=True를 사용하면 인덱스를 무시하고 0,1,2,3,4.... 이런식으로 채워진다.
      
      
      ```

      

  - merge 메서드

    ```python
    #merge 메서드는 기본적으로 내부 조인 실행
    #site를 왼쪽, visited_subset을 첫 번째 인자 / left_on, right_on은 값이 일치해야 할 왼쪽과 오른쪽 데이터프레임 열 지정.
    site.merge(visited_subset, left_on= 'name', right_on='site')
    ```

    

  - ebola.count()

  ```python
  	ebola = pd.read_csv('C:/doit_pandas-master/data/country_timeseries.csv')
  	ebola.count()
  	#ebola.count()를 하면 누락값이 아닌 값의 개수를 구함.
  ```


  - ebola.fillna(0)
    - fillna 메서드에 0을 대입하면 누락값을 0으로 변경함.
    - fillna 메서드는 처리해야 하는 데이터프레임의 크기가 매우 크고 메모리를 효율적으로 사용해야 하는 경우 자주 사용.

  - ebola.fillna(method='ffill' / 'bfill')


    - 누락값이 나타나기 전의 값으로 누락값이 변경
    - 누락값이 나타난 이후의 첫 번째 값으로 변경

  - ebola.interpolate()


    - 누락값 양쪽에 있는 값을 이용하여 중간값 계산


​    

  - ebola.dropna()


    - 누락값을 삭제 하기 위함.

- 깔끔한 데이터

  - melt 메서드 => 열이 너무 길 때 행으로 처리하기 위해 사용.

    - 지정한 열의 데이터를 모두 행으로 정리

    - 인자

      - id_vars

        - 위치를 그대로 유지할 열의 이름을 지정

      - var_name

        - value_vars로 위치를 변경한 열의 이름을 지정

      - value_name

        - var_name으로 위치를 변경한 열의 데이터를 저장한 열의 이름을 지정.

        ```pyth
        pew_long = pd.melt(pew, id_vars = 'religion', var_name = 'income', value_name = 'count'
        ```

  - 하나의 열에 둘의 의미가 있을 때.

    - str.split 메서드로 열 이름 분리하기

      ```python
      ebola_long['variable'].str.split('_')[:5]
      #str.split('_')로 사용.
      variable_split = ebola_long.variable.str.split('_', expand=True)
      #expand=True로 설정하면 데이터프레임으로 만들어짐.
      ```

      

    - str.get 분리된 데이터를 , 를 기준으로 출력.

      ```pyth
      variable_split = ebola_long.variable.str.split('_')
      status_values = variable_split.str.get(0)
      country_values = variable_split.str.get(1)
      ```

  - pivot_table 메서드

    - 행과 열의 위치를 다시 바꿔 정리함.

    - index인자 - 위치를 그대로 유지할 열 이름 지정

    - columns 인자 - 피벗할 열 이름 지정

    - values 인자 - 새로운 열의 데이터가 될 열 이름 지정

      ```python
      weather_melt = pd.melt(weather, id_vars=['id','year','month','element'])
      #melt로 id, year, month, element를 기준으로 pivot을 진행했지만, element가 tmax, tmin으로 나누어져 이것을 기준으로 다시 하고 싶다.
      
      weather_tidy = weather_melt.pivot_table(index = ['id', 'year', 'month', 'variable'], columns = 'element', values = 'value')
      #weather_melt.pivot_table 메서드는 행과 열의 위치를 다시 바꿔 정리.
      #index인자에는 위치를 그대로 유지할 열 이름을 지정.
      #columns인자에는 피벗할 열 이름을 지정(tmax, tmin으로 나뉨)
      #values인자에는 새로운 열의 데이터가 될 열 이름을 지정.
      ```

      

  - reset_index()

  - ```python
    weather_tidy_flat = weather_tidy.reset_index()
    #pivot_table까지 완성하고, reset_index()를 적용하면 예쁜 데이터프레임이 다시 완성된다.
    ```

    

    

- 자료형 변환

  - astype

    - tips['sex'].astype(str)
      - 카테고리 자료형을 문자형으로 변환

  - to_numeric

    ​	

    ```python
    pd.to_numeric(tips_sub_miss['total_bill'], errors = 'coerce', downcast='float')
    # error
    #raise - 숫자로 변환할 수 없는 값이 있으면 오류 발생.
    #coerce - 숫자로 변환할 수 없는 값을 누락값으로 지정.
    #ignore - 아무 작업도 하지 않음.(오류 발생x, 자료형도 변화 x)
    # downcast
    #float64 -> float32
    
    ```

    

- 문자열 메서드

  - 189쪽 참고

  - replace('fleshwound', 'scratch')

    - fleshwound를 scratch로 변환하라.

  - join, splitlines, replace 메서드 실습하기.

    - join메서드

      - 문자열을 연결하여 새로운 문자열을 반환하는 메서드

      ```python
      num1 = '010'
      num2 = '2112'
      num3 = '2914'
      total = '-'.join([num1,num2,num3])
      print(total)
      # -를 기준으로 num1, num2, num3를 연결한다.
      ```

    - splitlines 메서드

      - 여러 행을 가진 문자열을 분리한 다음 리스트로 반환

      ```python
      multi_str = """Gurad:afjklsdf
      1212323123
      sdfjklssdjfkl
      jvkdf
      """
      multi_str_split = multi_str.splitlines()
      print(multi_str_split)
      
      결과값
      ['Gurad:afjklsdf', '1212323123', 'sdfjklssdjfkl', 'jvkdf']
      ```

    - replace 메서드

      - 문자열을 치환해주는 메서드(위 참고)



- 숫자 데이터 포매팅하기

  - ```python
    "In 2005, Lu Chao of China recited {:,} digits of pi".format(12789127389)
    #{:,}를 사용하면 뒤에 세자리 형태로 끊는다.
    결과
    'In 2005, Lu Chao of China recited 12,789,127,389 digits of pi'
    ```

    

- apply메서드 활용하기

  - 사용자가 작성한 함수를 한 번에 데이터프레임의 각 행과 열에 적용하여 실행할 수 있게 해주는 메서드
  - = 함수를 브로드캐스팅해야 하는 경우에 사용.

  ```python
  #예시1 - 열 방향으로 데이터를 처리
  def df_avg_apply(col):
      sum = 0
      for item in col:
          sum += item
      return sum/df.shape[0]
  df.apply(df_avg_apply, axis=0)
  #결과
  a    20.0
  b    30.0
  dtype: float64
  
  
  #예시2 - 행 방향으로 데이터를 처리
  def df_avg_apply(col):
      sum = 0
      for item in col:
          sum += item
      return sum/df.shape[1]
  df.apply(df_avg_apply, axis=1)
  #결과
  0    15.0
  1    25.0
  2    35.0
  dtype: float64
  #예시3 - titanic 비율 누락값 / 누락값 비율 / 실제 데이터 비율 계산
  def count_missing(col):
      col_missing = col.shape[0] - col.count()
      return col_missing
  
  def prop_missing(col):
      num = count_missing(col)
      size = col.shape[0]
      return num / size
  
  def prop_complete(col):
      return 1 - prop_missing(col)
  
  titanic.apply(count_missing) # 누락값 계산
  titanic.apply(prop_missing) #누락값 비율 계산
  titanic.apply(prop_complete) # 실제 데이터 비율 계산
  #열로 했을 때 (axis = 0) 생략.
  
  axis=1로 두면 행으로 한다.
  
  ```

  

- 데이터 집계

  - groupby 메서드

  - 221쪽

  - 사용자 함수 + groupby 메서드

    ```python
    #예시1
    def my_mean(col):
        n = len(col)
        sum = 0
        for value in col:
            sum += value
        return sum / n
    
    df.groupby('year').lifeExp.agg(my_mean) #인자 1개일 때 -> 사용자 함수 활용.
    #예시2 - 사용자 정의 활용
    def my_mean_diff(values, diff_value):
        n = len(values)
        sum = 0
        for value in values:
            sum += value
        mean = sum / n
        return mean - diff_value
    
    df.groupby('year').lifeExp.agg(my_mean_diff, diff_value = global_mean)
    #예시3 - 집계 메서드를 적용할 열 이름 전달하고 값으로 집계 메서드 전달.
    df.groupby('year').agg({'lifeExp' : 'mean', 'pop' : 'median', 'gdpPercap' : 'median'})
    ```

    

- 데이터 변환

  - 정규화

  - ```python
    def my_zscore(x):
        return (x-x.mean())/x.std()
    
    transform_z = df.groupby('year').lifeExp.transform(my_zscore)
    ```

    

- 누락값을 평균값으로 처리하기.

  - ```python
    tips_10 = tips.sample(10)
    tips_10.loc[[37,172,204],['total_bill', 'tip']] = NaN
    #random값 10개 중 3개 인덱스 NaN으로 처리.
    
    def fill_na(col):
        avg = col.mean()
        return col.fillna(avg)
    
    tips_10['fill total_bill'] = tips_10.groupby('sex').total_bill.apply(fill_na)
    #성별로 구분하여 평균을 계산하여 fill_na로 채운다.
    
    tips_10['fill tip'] = tips_10.groupby('sex').tip.apply(fill_na)
    #성별로 구분하여 tip에 대한 평균을 fill_na에 대입하고 
    ```

- 그룹 오브젝트에서 데이터 추출 => get_group

  - ```python
    tips_10.groupby('sex').get_group('Male')
    #그룹 sex에서 Male만 추출하기.
    ```

  - 234쪽 확인

  

- datetime 오브제그로 변환하기 - to_datetime 메서드
  - datetime 오브젝트 d1의 year, month, day 속성을 이용하면 년, 월, 일 정보 바로 추출
    - d1[0].year => 2018
    - d1[0].month => 5
    - d1[0].day => 16

- dt 접근자 사용하기.

  - ```python
    ebola['month'], ebola['day'] = ebola['date_dt'].dt.month, ebola['date_dt'].dt.day
    #ebola['date_dt'].dt.month를 적용하면 열로 쭉 나옴.
    ```

  - 

- 실전 예제

  - 날짜 옮기기

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    ebola = pd.read_csv('C:/doit_pandas-master/data/country_timeseries.csv', parse_dates=['Date'])
    ebola.index = ebola['Date']
    new_idx = pd.date_range(start = ebola.index.min(), end= ebola.index.max())
    new_idx = reversed(new_idx)
    ebola = ebola.reindex(new_idx)
    #각 나라의 에볼라 발병일 옮기기
    #last_valid_index => 가장 오래된 데이터
    #first_valid_index => 가장 최근 데이터
    last_valid = ebola.apply(pd.Series.last_valid_index)
    first_valid = ebola.apply(pd.Series.first_valid_index)
    
    earliest_date = ebola.index.min()
    shift_values = last_valid - earliest_date
    
    ebola_dict={}
    for idx, col in enumerate(ebola):
        d = shift_values[idx].days
        shifted = ebola[col].shift(d)
        ebola_dict[col] = shifted
    ebola_shift = pd.DataFrame(ebola_dict)
    ebola_shift = ebola_shift.drop(['Date', 'Day'], axis=1)
    
    ax = ebola_shift.iloc[:,:].plot()
    ax.legend(fontsize=7)
    ```

    