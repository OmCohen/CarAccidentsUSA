#!/usr/bin/env python
# coding: utf-8

# 1.בחרנו בנתונים על תאונות דרכים בארהב.

# 2.מאגר הנתונים סוקר את התאונות דרכים בארה"ב ואת ההשפעות שלהם על התנועה,בנוסף הוא כולל נתונים מזג אוויר,מיקום.<br>
# הנתונים נאספו מ2016-2021 ובאמצעות דיווחי משטרה,רשויות שונות,מצלמות תנועה וסנסורים בכביש.<br>
#

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)

# In[2]:


df_Accident = pd.read_csv("US_Accidents_Dec21_updated.csv")

# In[3]:


df_Accident.info()

# ## הסבר על כל משתנה בנתונים:
#
# 1	ID	This is a unique identifier of the accident record.
# 2	Severity	Shows the severity of the accident, a number between 1 and 4, where 1 indicates the least impact on traffic (i.e., short delay as a result of the accident) and 4 indicates a significant impact on traffic (i.e., long delay).<br>
# 3	Start_Time	Shows start time of the accident in local time zone.
# 4	End_Time	Shows end time of the accident in local time zone. End time here refers to when the impact of accident on traffic flow was dismissed.
# 5	Start_Lat	Shows latitude in GPS coordinate of the start point.
# 6	Start_Lng	Shows longitude in GPS coordinate of the start point.
# 7	End_Lat	Shows latitude in GPS coordinate of the end point.
# 8	End_Lng	Shows longitude in GPS coordinate of the end point.
# 9	Distance(mi)	The length of the road extent affected by the accident.
# 10	Description	Shows natural language description of the accident.
# 11	Number	Shows the street number in address field.
# 12	Street	Shows the street name in address field.
# 13	Side	Shows the relative side of the street (Right/Left) in address field.
# 14	City	Shows the city in address field.
# 15	County	Shows the county in address field.
# 16	State	Shows the state in address field.
# 17	Zipcode	Shows the zipcode in address field.
# 18	Country	Shows the country in address field.
# 19	Timezone	Shows timezone based on the location of the accident (eastern, central, etc.).
# 20	Airport_Code	Denotes an airport-based weather station which is the closest one to location of the accident.
# 21	Weather_Timestamp	Shows the time-stamp of weather observation record (in local time).
# 22	Temperature(F)	Shows the temperature (in Fahrenheit).
# 23	Wind_Chill(F)	Shows the wind chill (in Fahrenheit).
# 24	Humidity(%)	Shows the humidity (in percentage).
# 25	Pressure(in)	Shows the air pressure (in inches).
# 26	Visibility(mi)	Shows visibility (in miles).
# 27	Wind_Direction	Shows wind direction.
# 28	Wind_Speed(mph)	Shows wind speed (in miles per hour).
# 29	Precipitation(in)	Shows precipitation amount in inches, if there is any.
# 30	Weather_Condition	Shows the weather condition (rain, snow, thunderstorm, fog, etc.)
# 31	Amenity	A POI annotation which indicates presence of amenity in a nearby location.
# 32	Bump	A POI annotation which indicates presence of speed bump or hump in a nearby location.
# 33	Crossing	A POI annotation which indicates presence of crossing in a nearby location.
# 34	Give_Way	A POI annotation which indicates presence of give_way in a nearby location.
# 35	Junction	A POI annotation which indicates presence of junction in a nearby location.
# 36	No_Exit	A POI annotation which indicates presence of no_exit in a nearby location.
# 37	Railway	A POI annotation which indicates presence of railway in a nearby location.
# 38	Roundabout	A POI annotation which indicates presence of roundabout in a nearby location.
# 39	Station	A POI annotation which indicates presence of station in a nearby location.
# 40	Stop	A POI annotation which indicates presence of stop in a nearby location.
# 41	Traffic_Calming	A POI annotation which indicates presence of traffic_calming in a nearby location.
# 42	Traffic_Signal	A POI annotation which indicates presence of traffic_signal in a nearby loction.
# 43	Turning_Loop	A POI annotation which indicates presence of turning_loop in a nearby location.
# 44	Sunrise_Sunset	Shows the period of day (i.e. day or night) based on sunrise/sunset.
# 45	Civil_Twilight	Shows the period of day (i.e. day or night) based on civil twilight.
# 46	Nautical_Twilight	Shows the period of day (i.e. day or night) based on nautical twilight.
# 47	Astronomical_Twilight	Shows the period of day (i.e. day or night) based on astronomical twilight.
# (לקוח מהאתר קגל)

# בחרנו להוריד את המשתנים הבאים שכן הם לא רלוונטים להמשך המחקר או ספציפיים מידי.
#

# In[4]:


df_Accident_cleaned = df_Accident.drop(
    columns=["Start_Time", "End_Time", "Start_Lat", "Start_Lng", "End_Lat", "End_Lng", "Distance(mi)",
             "Description", "Number", "Street", "Zipcode", "Country", "Airport_Code", "Weather_Timestamp",
             "Wind_Chill(F)", "Pressure(in)",
             "Wind_Direction", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"])

# In[5]:


df_Accident_cleaned.info()

# אלו הן השורות הרלוונטיות שבחרנו להשאיר.

# In[6]:


df_Accident_cleaned.notnull().count()

# ## הצגת פילוג של נתונים

# נציג את הפילוג של הערכים בעמודת State<br>
# כלומר, הפילוג של התאונות לפי מדינות בארה"ב

# In[7]:


stateFig = sns.catplot(data=df_Accident_cleaned, x="State", kind="count")
stateFig.fig.set_size_inches(15, 10)
stateFig.set(xlabel="State(initial)", ylabel="Car Accidents")
stateFig.fig.suptitle("Number of car accidents distributed by State in the US(2016-2021)")

# ### מסקנות מגרף מספר 1
# ניתן לראות כי רוב התאונות מתרחשות בפלורידה וקליפורניה,אחריה אורגון וטקסס

# In[8]:


elements = pd.DataFrame()
elements['Element'] = ['Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
                       'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
elements['Car_Accidents'] = elements['Element'].apply(
    lambda x: df_Accident_cleaned[df_Accident_cleaned[x] == True][x].count())

# In[9]:


elements

# In[10]:


crossing_Fig = sns.catplot(data=elements, x='Element', y='Car_Accidents', kind="bar");
crossing_Fig.fig.suptitle("most accidents near those elements")
crossing_Fig.fig.set_size_inches(12, 6)
crossing_Fig.set_xticklabels(rotation=45)
crossing_Fig.set(ylabel="Car Accidents")

# ### מסקנות מגרף מספר 2
# אנו למדים מהגרף שמבין הגורמים השונים הנמצאים (בכביש ועליהם יש לנו מידע) מספר התאונות הרבה ביותר מתרחש סביב צמתים, מעברי חציה ורמזורים.<br>
# המשותף לגורמים אלו הינו השתלבות של כלי רכב שונים או כלי רכב והולכי רגל ואנו מאמינים שהשתלבויות כאלו מעלות את הסיכון לתאונות.<br>
# אנו בכל זאת מסתייגים שכן תמרורים כגון "עצור" ו"תן זכות קדימה" אינם גורמים חזקים לתאונות באופן יחסי בדאטה וגם שם ישנם מצבים של השתלבויות בשדות ראיה מוגבלים.

# In[11]:


df_Accident_cleaned[df_Accident_cleaned["Weather_Condition"].isna()].count()

# ישנם 70 אלף רשומות בהם אין תיאור מזג אוויר. אנו מעריכים שהדבר נגרם מטעות מדידה של אי דיווח של הגורמים המעורבים ולא משפיע על עצם קיום התאונה. לכן נוריד רשומות אלו.
# גודל הנתונים כולל כמעט 3 מליון רשומות לכן יחסית הדבר זניח.

# In[12]:


df_Accident_cleaned_and_weather = df_Accident_cleaned[df_Accident_cleaned["Weather_Condition"].isna() == False]
df_Accident_cleaned_and_weather[df_Accident_cleaned_and_weather['Weather_Condition'].isna()].count()

# ישנם רשומות רבות לגבי מזגי אוויר ספציפים,שכן המידע נאסף כתיאור של אנשים בסביבה והם לא עקביים,לכן נאגד אותם לפי סוג מזג האוויר ביותר כלליות.בחרנו לחלק לפי רמות חומרה של מזג האוויר ולכן יש חשיבות לסדר.

# In[13]:


sunny_Weather = np.array(['Clear', 'Fair', 'N/A Precipitation'])
rain_Weather = np.array(
    ['Light Rain', 'Rain', 'Light Drizzle', 'Light Freezing Drizzle', 'Heavy Rain', 'Drizzle', 'Thunderstorms and Rain'
                                                                                               'Light Thunderstorms and Rain',
     'Rain Showers', 'Light Rain Showers', 'Heavy Drizzle', 'Light Freezing Rain',
     'Heavy Thunderstorms and Rain', 'Thunderstorm', 'Light Rain with Thunder', 'Rain / Windy', 'Light Rain / Windy',
     'Freezing Rain / Windy', 'Small Hail', 'Wintry Mix / Windy', 'Heavy Rain / Windy', 'Showers in the Vicinity',
     'Rain Shower', 'Light Snow Shower', 'Freezing Rain', 'Heavy Freezing Drizzle', 'Heavy Freezing Rain',
     'Heavy Rain Shower',
     'Heavy Thunderstorms with Small Hail'])
low_visability_Weather = np.array(
    ['Mist', 'Haze', 'Smoke', 'Light Freezing Fog', 'Shallow Fog', 'Sand', 'Widespread Dust',
     'Blowing Dust', 'Volcanic Ash', 'Fog / Windy', 'Sand / Dust Whirlwinds', 'Drizzle and Fog',
     'Widespread Dust / Windy', 'Sand / Dust Whirlwinds / Windy', 'Sand / Dust Whirls Nearby'])
hard_Weather = np.array(
    ['Snow', 'Blowing Snow', 'Heavy Snow', 'Heavy Snow / Windy', 'Snow and Sleet', 'Heavy Sleet', 'Blowing Sand',
     'Heavy Blowing Snow''Heavy Snow with Thunder'])


def weather_counter(df):
    sunny = 0
    rain = 0
    low = 0
    hard = 0
    for i in df["Weather_Condition"].values:
        if i in sunny_Weather:
            sunny += 1
        if i in rain_Weather:
            rain += 1
        if i in low_visability_Weather:
            low += 1
        if i in hard_Weather:
            hard += 1
    return (sunny, rain, low, hard)


# In[14]:


(sunny, rain, low, hard) = weather_counter(df_Accident_cleaned)
d = {'type': ["Sunny_Weather", "Rain_Weather", "Low_visability_Weather", "Hard_Weather"],
     'amount': [sunny, rain, low, hard]}
df_Weather = pd.DataFrame(data=d)
df_Weather.head()

# In[15]:


Weather_Fig = sns.catplot(x="type", y="amount", kind="bar", data=df_Weather)
Weather_Fig.fig.suptitle("number of accidents by weather condition")
Weather_Fig.fig.set_size_inches(8, 6)

# ### מסקנות מגרף מספר 3
# ניתן לראות כי רוב התאונות הן בתנאי מזג אוויר טובים,אך גם יש הרבה יותר ימים כאלה,אנו רצינו רק לראות כמויות ובמידה והיו לנו נתונים על מספר הימים בהם היו תנאי מזג אוויר "מיוחדים" היינו משתמשים בכך לנרמל את הדאטה כדי להגיע למסקנות מדויקות יותר

# In[16]:


df_Accident_cleaned[df_Accident_cleaned["Temperature(F)"].isna()].count()

# נשים לב שיש בערך 69 אלף שורות חסרות בעמודת הטמפרטורה. גם במקרה זה אנו מעריכים כי מדובר על הטיית מדידה הקשורה לאיכות מדידת הטמפרטורה ואינה קשורה לקיום או אי קיום התאונה(אלא לגורמים באתר מדידת מזג האוויר הקרוב). לכן נוריד שורות אלו

# In[17]:


df_Temperature = df_Accident_cleaned["Temperature(F)"].dropna()
df_Temperature.count()

# In[18]:


df_Temperature

# In[19]:


df_Temperature = df_Temperature.apply(lambda x: ((x - 32) * 5) / 9)  # convert to Celsius

# In[20]:


g = sns.catplot(data=df_Temperature.to_frame(), y="Temperature(F)", kind="box", height=4, aspect=.7)
g.fig.set_size_inches(8, 6)
g.set(ylabel="Temperature(C)")
g.fig.suptitle("Temperatures measured when car accidents occurred")

# In[21]:


print(f"the mean is:{df_Temperature.mean()}")
# we'll use the quantiles to understand how our data is distributed
Q1 = df_Temperature.quantile(0.25)
Q3 = df_Temperature.quantile(0.75)
IQR = Q3 - Q1
print(f"The Q1 is:{Q1}")
print(f"The Q3 is:{Q3}")
print(
    f"The IQR is:{IQR}")  # from the IQR we can see that the scope of temperatures is rather wide but still around comfortable conditions

# ### מסקנות מגרף מספר 4
# נשים לב שישנה התפלגות יחסית נורמאלית סביב טמפרטורה של 16 מעלות שהינה מזג אוויר נעים.<br>
# בנוסף נראה כי ממוצע הערכים הוא סביב טמפרטורות נוחות ולכן במובן זה רוב התאונות מתרחשות בתנאי מזג אוויר רגילים ולא קיצוניים<br>
# תופעה זו, כמו גם התופעה בגרף הקודם כנראה נגרמות מכך שיש הרבה יותר ימים "רגילים" בשנה מימים "קיצוניים" מבחינת מזג אוויר אך תאונות דרכים אינן תופעה מיוחדת אלא תופעה שגרתית לצערנו

# In[22]:


dayNightGraph = sns.catplot(kind="count", data=df_Accident_cleaned, x="Sunrise_Sunset", palette="rocket")
dayNightGraph.fig.suptitle("Number of car accidents based on time of day")
dayNightGraph.set(ylabel="Car accidents")
dayNightGraph.fig.set_size_inches(6, 5)

# In[24]:


print(f"The precentage of night accidents are:{1030540 * 100 / (2842475)}%")

# In[25]:


print(f"The precentage of day accidents are:{(1811935 / (2842475)) * 100}%")

# ### מסקנות מגרף מספר 5
# נשים לב כי רוב התאונות מתרחשות ביום.<br>
# אנו מעריכים שזה נגרם מכך שרוב התנועה הינה במהלך שעות היום ולמרות זאת חלק לא מבוטל של 36 אחוז מהתאונות מתרחש בשעות הלילה

# ## 2.b. bivariate relationships

# In[26]:


## א


# ננסה להראות את הקשר בין מספר התאונות לגודל  האוכלוסייה

# נשתמש בנתונים על גודל האוכלוסיה במדינה(state)<br>
# את הנתונים הללו לקחנו מהאתר: https://data.ers.usda.gov/reports.aspx?ID=17827#P1441838c27b242998ceb565731a2bf78_2_153iT3<br>
# בחרנו באתר זה שכן זה אתר ממשלתי ולכן אנו סבורים שהנתונים יהיו אמינים. הנתונים מעודכנים נכון ליוני 2022

# In[27]:


df_popualtion = pd.read_csv("us-states-territories.csv")

# In[28]:


df_population_clean = df_popualtion[["Abbreviation", "Population (2019)"]].dropna()
df_population_clean.head()

# המרנו את מספר התושבים שהיה מחרוזת למספר(int)

# In[29]:


df_population_clean["Population (2019)"] = df_population_clean["Population (2019)"].apply(lambda x: x.split(","))
df_population_clean["Population (2019)"] = df_population_clean["Population (2019)"].apply(lambda x: int("".join(x)))

# In[30]:


df_population_clean.head()

# In[31]:


df_States = df_Accident_cleaned.groupby("State").agg(Car_Accidents=("ID", "count")).reset_index()
df_States.head()

# המאגר נתונים היה מלא ברווחים לכן הורדנו את הרווחים

# In[32]:


df_population_clean["Abbreviation"] = df_population_clean["Abbreviation"].apply(lambda x: x.replace(" ", ""))

# In[33]:


df_population_clean.head()

# נחבר בין שני המאגרים

# In[34]:


df_States["Population"] = df_States["State"].apply(
    lambda x: df_population_clean[df_population_clean["Abbreviation"] == x].values[0, 1])
df_States.head()

# In[35]:


df_States.sort_values("Population", inplace=True)
df_States.reset_index(inplace=True)

# In[36]:


accident_Population = sns.relplot(data=df_States, x='State', y='Car_Accidents', kind="line")
accident_Population.fig.set_size_inches(13, 7)
accident_Population.set_xticklabels(rotation=45)
accident_Population.set(xlabel="State initials(low population to high population)", ylabel="Car accidents")
accident_Population.fig.suptitle("Number of car accidents in a state in relation to population in the state.")

# In[37]:


df_States[["Car_Accidents", "Population"]].corr(method='pearson')

# ### מסקנות מגרף מספר 6
# אנו רואים מהגרף כי יש קורלציה,זאת מכיוון כי יש מגמה של גידול באוכלוסייה לעומת עלייה בתאונות נשים לב שלעיתים יש ירידה <br>
# אבל נוכל לראות ממקדם הקורלציה שחושב על ידי נוסחאת פירסון שהמקדם חיובי ואפילו גבוה(0.83) ולכן יש אפשר להעריך כי קיימת קורלציה.

# In[38]:


## ב


# בגרף מספר 2 ראינו ששלושת האלמנטים סביבם היו הכי הרבה תאונות הם: רמזורים, צמתים, מעברי חצייה.<br>
# הערכנו שהמשותף הינו השתלבות של כלי רכב והולכי רגל ושדות ראייה מוגבלים. מצבים כאלו יכולים לגרום לבלימות חירום, תגובות ברגע האחרון ואף תאונות.<br>
# לכן יכולה להיות קורלציה בין תאונות סביב גורמים אלו ותנאי מזג אוויר קשים.<br>
# נסתכל על סוג תנאי מזג האוויר כמשתנה קטגוריאלי אורדינלי(לפי סדר הקושי: שמשי, ראות לקויה, גשום, תנאי מזג אוויר קשים) וננסה להעריך האם יש קורלציה בין החמרה בתנאי מזג האוויר לבין כמות התאונות סביב אלמנטים אלו בכביש.

# ניצור שורה המכילה את המידע עבור כל אלמנט ולבסוף נחבר לטבלה אחת

# In[39]:


crossing_filterd = df_Accident_cleaned[df_Accident_cleaned["Crossing"] == True]
crossing_filterd.head()

# In[40]:


(sunny, rain, low, hard) = weather_counter(crossing_filterd)
d = {'sunny': [sunny], "low visibility": [low], 'rain': [rain], "hard weather": [hard]}
df_Crossing_Filterd = pd.DataFrame(data=d)
df_Crossing_Filterd['Element'] = "Crossing"
df_Crossing_Filterd.head()

# In[41]:


junction_filterd = df_Accident_cleaned[df_Accident_cleaned["Junction"] == True]
junction_filterd.head()
(sunny, rain, low, hard) = weather_counter(junction_filterd)
d = {'sunny': [sunny], "low visibility": [low], 'rain': [rain], "hard weather": [hard]}
df_junction_filterd = pd.DataFrame(data=d)
df_junction_filterd['Element'] = "Junction"
df_junction_filterd.head()

# In[42]:


traffic_signal_filterd = df_Accident_cleaned[df_Accident_cleaned["Traffic_Signal"] == True]
traffic_signal_filterd.head()
(sunny, rain, low, hard) = weather_counter(traffic_signal_filterd)
d = {'sunny': [sunny], "low visibility": [low], 'rain': [rain], "hard weather": [hard]}
df_traffic_signal = pd.DataFrame(data=d)
df_traffic_signal['Element'] = "Traffic Signal"
df_traffic_signal.head()

# In[43]:


combined_element_df = df_traffic_signal.append(df_junction_filterd)
combined_element_df = combined_element_df.append(df_Crossing_Filterd)
combined_element_df.reset_index(inplace=True)
combined_element_df.drop(columns="index", inplace=True)
combined_element_df

# In[44]:


fig, axs = plt.subplots()
combined_element_df.plot(kind="bar", x="Element", y=["sunny", "low visibility", "rain", "hard weather"], ax=axs)
fig.set_size_inches(8, 5)
fig.suptitle("Number of car accidents per weather type that occured next to a given element")
axs.set_ylabel("Car Accidents")
axs.set_xticklabels(axs.get_xticklabels(), rotation=45)

# גם כעת אנו מעריכים שמספר הימים השמשיים כה משמעותי ביחס לשאר שכמות התאונות בו גם גבוה יותר ולא בהכרח כי הסיכון גבוה. נתעלם מתאונות בימים אלו וננסה להסתכל על המגמה בימים בהם תנאי מזג האוויר לא אופטימאלים מסיבות שונות(ראות לקויה, כביש חלק, רוחות חזקות וכו)

# In[45]:


fig, axs = plt.subplots()
combined_element_df.plot(kind="bar", x="Element", y=["low visibility", "rain", "hard weather"], ax=axs)
fig.set_size_inches(8, 5)
fig.suptitle("Number of car accidents per weather type that occured next to a given element")
axs.set_ylabel("Car Accidents")
axs.set_xticklabels(axs.get_xticklabels(), rotation=45)

# ### מסקנות מגרף מספר 7
# לא נוכל להעריך קורלציה בין עלייה בחומרה של מזג האוויר לתאונות סביב גורמים בעייתים בכביש(רמזורים, צמתים ומעברי חציה).<br>
# לפי הגרף נראה כי בשלושת המקרים אין קורלציה אך אנו מסתייגים מקביעת אי הקורלציה שכן חסרים לנו נתונים על כמות הימים מכל סוג במהלך השנה עצמה ולכן לא נוכל לנרמל את הנתונים שלנו.<br>
# בכל זאת נראה כי כמות התאונות תחת מזג אוויר גשום הינה גבוה יותר סביב צמתים מאשר סביב אלמנטים אחרים ובפרט יותר מאשר בצמתים מרומזרים.

# # חלק ג' - בחינת השערות

# אנחנו משערים שמספר התאונות לנפש בקליפורניה הוא חריג,נשתמש בנתונים שהבאנו מהמשרד פנים האמריקאי לגבי גודל האוכלוסייה וכמובן במאגר הנתונים שלנו בשנים 2016-2021

# השערת האפס שלנו היא כי אין הבדל בין שיעור התאונות לנפש בקליפורניה לבין שיעור התאונות לנפש הכללי בארצות הברית(בלי קליפורניה),נשער שההבדל הוא 0 אחוז.

# ההשערה האלטרניטיבית היא כי יש הבדל בשיעור התאונות בקליפורניה לנפש לבין שאר(ההפרש לא שווה אפס)

# נתחיל מייצרת מסד נתונים לתאונות בקליפורניה ותאונות בשאר ארה"ב וחישוב של סטטיסטי המבחן שלנו(הפרש אחוזי התאונות לנפש)

# In[46]:


df_without_cali = df_Accident_cleaned[df_Accident_cleaned["State"] != "CA"]
population_without_cali = df_States[df_States["State"] != "CA"]["Population"].sum()
car_accidents_without_cali = df_without_cali.shape[0]
car_accidents_without_cali

# In[47]:


accident_ratio_no_cali = car_accidents_without_cali / population_without_cali
accident_ratio_no_cali

# In[48]:


df_cali = df_Accident_cleaned[df_Accident_cleaned["State"] == "CA"]
accident_ratio_in_cali = df_cali.shape[0] / df_Accident_cleaned.shape[0]

# In[49]:


accident_ratio_in_cali

# In[85]:


print(f" The test statistic is: {accident_ratio_in_cali - accident_ratio_no_cali}")


# זהו ניסיון הבוטסטראפ שלא צלח

# In[50]:


def bootstrap_diff(original_sample, column_name, num_replications, population_size_cali, population_size_not_cali):
    original_sample_size = original_sample.shape[0]
    original_sample_var_of_interest = original_sample[[column_name]]
    bstrap_diff = np.empty(num_replications)
    for i in range(num_replications):
        bootstrap_sample = original_sample_var_of_interest.sample(original_sample_size, replace=True)
        cali_percentage = bootstrap_sample[bootstrap_sample["State"] == "CA"].count() / population_size_cali
        not_cali_percentage = bootstrap_sample[bootstrap_sample["State"] != "CA"].count() / population_size_not_cali
        bstrap_diff[i] = cali_percentage - not_cali_percentage
    return bstrap_diff


# In[51]:


# df_bootstrap = df_Accident_cleaned[["State","ID"]]


# In[52]:


# bootstrap = bootstrap_diff(df_bootstrap, "State", 1000, df_cali.shape[0], df_without_cali.shape[0])
# bootstrap


# ההרצה של הבוטסטראפ כמעט ואיננה אפשרית עבור כמות כזו גדולה של הרצות. נניח כי השערת האפס נכונה ונפלג את הפרשי התאונות לנפש לפיה. נשווה את סטטיסטי המבחן להתפלגות זו

# In[53]:


precentege_in_population_cali = df_States[df_States["State"] == "CA"]["Population"].sum() / df_States[
    "Population"].sum()
accident_place = ['california', 'rest']
prob_for_item = [precentege_in_population_cali, 1 - precentege_in_population_cali]
sample_size = df_Accident_cleaned.shape[0]


# simulate one value
def prob_accident_in_cali():
    sample_Us_Accident = np.random.choice(accident_place, p=prob_for_item, size=sample_size)
    num_cali = np.count_nonzero(sample_Us_Accident == 'california')
    return num_cali / sample_size


# run multiple simulations
num_repetitions = 1500
many_prob_cali = np.empty(num_repetitions)  # collection array
for i in range(num_repetitions):
    many_prob_cali[i] = prob_accident_in_cali()

# In[54]:


facetgrid_obj = sns.displot(many_prob_cali, stat='probability')
facetgrid_obj.fig.set_size_inches(10, 7)
facetgrid_obj.set(title='precenteage of california accident in USA by population',
                  xlabel='proportion of accident from cali in precentage', ylabel='num of simulations')

# כאן הנחנו כי השערת האפס נכונה ואכן אחוז התאונות בקליפורניה תואם לגודל האוכלוסייה.

# In[55]:


facetgrid_obj = sns.displot(many_prob_cali, stat='probability')
facetgrid_obj.fig.set_size_inches(10, 7)
facetgrid_obj.set(title='', xlabel='', ylabel='')
facetgrid_obj.axes[0, 0].scatter(accident_ratio_in_cali, 0, s=150, color='red')  # draw observed value
facetgrid_obj.axes[0, 0].legend(['Test statistic'])
facetgrid_obj.set(title='precenteage of california accident in USA by population',
                  xlabel='proportion of accident from cali in precentage', ylabel='num of simulations')

# כאן ניתן לראות את אחוז התאונות בקליפורניה מתוך כלל התאונות במדגם(מסומן באדום)

# In[56]:


greater_or_equal = np.count_nonzero(many_prob_cali >= accident_ratio_in_cali)
print(f'the p-value is {greater_or_equal / num_repetitions}')

# נראה כי הסטטיסטי מספק ערך 0 ולכן ניתן לדחות את השערת האפס(לא להיכנס לקליפורניה עם רכב)

# נגיד כי לא כל התאונות בקליפורניה הם של תושבי קליפורניה ויכול להיות שיש השפעה מתיירות פנים,עדיין ביחס לאוכלוסייה היחס קיצוני מאוד.

# ## 4 - Classification

#  ננסה לאמן מסווג שיצליח לזהות את חומרת התאונה(בהשפעה על התנועה בכביש) <br>
#  המטרה שלנו היא למצוא מסווג עבור כוח אדם משטרתי וגם להשתמש באותו מסווג עבור אפליקצית ניווט.

# עבור הניווט זה שימושי בשביל לדעת איך להכווין בהינתן שתאונה קרתה

# עבור המשטרה זה שימושי בשביל לדעת אם לשלוח כוחות,כמה ולמה להערך

# In[57]:


df_Accident_cleaned.info()

# נוריד עמודות שבוודאות אין להם רלוונטיות <br>
# בחרנו להוריד מזהה של התאונה כי הוא יחידי ולא באמת משפיע <br>
# שאר הנתונים הורדנו בגלל שהם ספצפיים למדינה,דבר אשר לא רלונטי למטרתנו,אנחנו רוצים להסתכל על תאונות בכללי ולהבין מה גורם לעומס שלהם.<br>
# בנוסף, נתונים כמו אורך זמן פינוי התאונה שמופיעים במסד הנתונים הכולל לא יהיו זמינים למסווג בעת ניסינו לסווג תאונה כחמורה ולכן גם עליהם נוותר
#

# In[58]:


df_Accident_class = df_Accident_cleaned.drop(columns=["Timezone", "ID", "County", "City", "State", "Weather_Condition"])
df_Accident_class = pd.get_dummies(df_Accident_class, columns=['Side', 'Sunrise_Sunset', 'Junction',
                                                               'No_Exit', 'Railway', 'Roundabout',
                                                               'Station', 'Stop', 'Traffic_Calming',
                                                               'Traffic_Signal',
                                                               'Turning_Loop', 'Amenity', 'Bump', 'Crossing',
                                                               'Give_Way'], drop_first=True)
df_Accident_class

# נשים לב כי יש נתון בעייתי בעמודת הצד של הכביש לכן נוריד את הרשומה הספציפית.
#

# In[59]:


myfilter = (df_Accident_cleaned["Side"] == "N")
df_Accident_cleaned[myfilter]

# In[60]:


df_Accident_cleaned.drop(index=1084708, inplace=True)


# נגדיר תאונה חמורה כתאונה שיש לה ערך Severity<br>
# של 3 ומעלה

# In[62]:


def under3(test):
    if test <= 2:
        return 0
    return 1


df_Accident_class["Severity"] = df_Accident_class["Severity"].apply(under3)

# In[63]:


df_Accident_class.isna().sum()

# נוריד את עמודת המשקעים בשל החוסר הגדול בנתונים(חצי מליון) בשאר הפרמטרים נוריד רק את השורות החסרות

# In[64]:


df_Accident_class.drop(columns="Precipitation(in)", inplace=True)

# In[65]:


df_Accident_class.dropna(inplace=True)

# In[66]:


df_Accident_class.isna().sum()

# נחפש פיצ'רים בעלי קורלציה גבוה לSeverity

# In[67]:


correlations = df_Accident_class.corr()
plt.figure(figsize=(20, 20))
g = sns.heatmap(correlations, annot=True, cmap="RdYlGn")

# בגלל שאנו מנסים לסווג את חומרת התאונה(בהשפעה על התנועה בכביש),אבל אנו רואים שאין קורלציה גבוהה למשתנה זה מאף נתון אחר,אז נאמן את המסווג לפי נתונים שאנו חושדים שישפיעו לפי ידע אישי שלנו<br>
# נתונים אלו הם:<br>
# 1.הימצאות צומת (צומת דורשת תיאום בין הרבה כלי רכב לכן מעלה סיכון להחמרה בפפק במידה והתאונה קרובה)<br>
# 2.ראות (ראות לא טובה מקשה על תפקוד המשטרה בשטח ומורידה את מהירות התגובה של נהגים אחרים על הכביש)<br>
# 3.יום\לילה (לילה יש פחות נהגים על הכביש,לכן אנו חושדים שיהיו הפקקים יהיו פחות חמורים)

# In[68]:


df_Accident_class.info()

# In[69]:


df_Accident_class = df_Accident_class.drop(columns=["Temperature(F)", "Wind_Speed(mph)", "Side_R", "No_Exit_True",
                                                    "Railway_True", "Roundabout_True", "Station_True",
                                                    "Stop_True", "Traffic_Calming_True", "Traffic_Signal_True",
                                                    "Amenity_True", "Give_Way_True", "Bump_True", "Crossing_True",
                                                    "Humidity(%)"])

# In[70]:


df_Accident_class.info()

# נחלק את הנתונים לסט אימון וסט בחינה(ננרמל גם את הרשומות). מכיוון ומסד הנתונים שלנו גדול מדי בכדי להריץ עליו את האלגוריתמים הבאים ולאמן את המסווג(ניסינו להריץ מספר פעמים ללא הצלחה... עד שהגענו למסקנה שהרצה של 300 אלף תצפיות היא המקסימלית שהמחשבים שלנו יצליחו להריץ) נשלוף מתוך המאגר מדגם קטן יותר באופן ראנדומלי וננסה לאמן את המסווג עליו. לכן נקבע כי המדגם המוקטן יהיה 10 אחוז(300 אלף תצפיות) מכלל המדגם.<br>
# לפני כן נוודא את כמות התאונות ה"חמורות" וכמות התאונות ה"לא חמרות" כדי להבטיח שיהיו מספיק דוגמאות מכל סוג בסט האימון

# In[71]:


df_accident_test = df_Accident_class.groupby("Severity")
df_accident_test.describe()

# ניתן לראות כי בהחלט קיים סיכון למצב בו אין תצפית של תאונה חמורה במדגם המוקטן כלל ולכן ניצור מדגם מוקטן באופן ידני על ידי דגימות משתי הקבוצות באופן פרופורציונאלי לגדלים שלהן(בערך 10 אחוז חמורות ו90 אחוז לא חמורות). ניצור מדגם מוקטן של 300,000 תצפיות כך שנוכל להריצו

# In[72]:


severity_0_sample = df_accident_test.get_group(0).sample(round((90 / 100) * 300000), replace=True)
severity_1_sample = df_accident_test.get_group(1).sample(round((10 / 100) * 300000), replace=True)

# In[73]:


comb_sample = pd.concat([severity_0_sample, severity_1_sample])
comb_sample.reset_index(inplace=True)
comb_sample.drop(columns="index", inplace=True)
comb_sample

# In[74]:


X = comb_sample.iloc[:, 1:]  # features
Y = comb_sample.iloc[:, 0]  # labels
# since the data set is huge we can't run the k-fold algorithm with a big train set.
# we will use a small train set(still containing a massive 600,000 entries) and a big test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

# In[75]:


scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# נריץ אלגוריתם K-Fold<br>
# כדי למצוא k אופטימלי

# In[76]:


k_results = np.zeros(21)
for k in range(1, 22):
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_div = cross_val_score(knn, scaled_X_train, Y_train, cv=5)
    k_results[k - 1] = cv_div.mean()

# In[77]:


k_fig, ax = plt.subplots()
ax.plot(k_results)
ax.set_xticks(np.arange(0, 21))
k_fig.suptitle("K nearest method performance using cross val of 5")
k_fig.set_size_inches(10, 10)
ax.scatter(np.argmax(k_results), np.max(k_results))
ax.set_ylim(0, 1)

# ניקח את הקיי הטוב ביותר

# In[78]:


best_k = np.argmax(k_results) + 1
best_k

# In[86]:


final_knn = KNeighborsClassifier(n_neighbors=best_k)  # best K
final_knn.fit(scaled_X_train, Y_train)
k_predict = final_knn.predict(scaled_X_test)
k_score = final_knn.score(scaled_X_test, Y_test)
k_predict

# In[87]:


cmtx = pd.DataFrame(
    confusion_matrix(Y_test, k_predict, labels=[0, 1]),
    index=['real: 0', 'real: 1'], columns=['pred: 0', 'pred: 1']
)
cmtx

# In[88]:


print(f"the accuracy is:{accuracy_score(Y_test, k_predict)}")

# In[89]:


print(f"the precision is:{precision_score(Y_test, k_predict)}")

# In[90]:


print(f"the recall is:{recall_score(Y_test, k_predict)}")

# In[91]:


print(f"the f1 score:{f1_score(Y_test, k_predict)}")

# חשבנו על שתי מטרות אפשריות לשימוש במסווג<br>
# 1.אפליקציה לניווט כמו וויז וגוגל מפות <br>
# 2.משטרה או רכבים לפינוי פקקים

# אילו אנו מסתכלים מהיבט משטרתי חשוב לנו להסתכל יותר על הדיוק שכן חשוב לנו לנצל נכון את כוח האדם ולא לבזבז קריאות ומשאבים

# אילו אנו מסתכלים מהיבט של אפליקציה לדוגמא היינו רוצים להסתכל יותר על הריקול ולא לפספס פקקים אפילו במחיר של לכוון לא מאותו כביש למרות שהאירוע הסתיים מהר

# סה"כ אנו יכולים לראות כי המסווג אינו מוצלח מספיק לאף מטרה(אם כי יותר טוב עבור המטרה המשטרתית בגלל שהדיוק גבוה יותר מהריקול). ייתכן והדבר נגרם מכך שמראש ישנם משמעותית יותר תאונות לא חמורות במדגם שלנו(בערך 90 אחוז מתוכו) ולכן המסווג נוטה לסווג את התצפיות כלא חמורות.<br>
# דבר הנתמך על ידי ערך הaccuracy הגבוה <br>
# בנוסף, כפי שראינו קודם וציפינו העובדה שאין קורלציה גבוה בין אף ערך לחומרת התאונה גם כן עלולה לגרום למסווג להתקשות מאוד.
# כרעיון לניסוי עתידי היינו מנסים לשפר את ביצועי המסווג על ידי השגת נתונים נוספים שיהיו בקורלציה גבוה יותר. נתונים שאנו חושדים שיעזרו הם: כמות\הימצאות נפגעים, האם התרחש על כביש ראשי או עירוני, מספר הרכבים המעורבים בתאונה וכו... נתונים אלו כנראה נוכל להשיג רק מדוחות משטרה(שלהם אין לנו גישה כעת) ולכן אנו לא יכולים להשתמש בהם

# In[ ]:




