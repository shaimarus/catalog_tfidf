from flask import Flask, request, render_template
import pickle
import time
from pymorphy2 import MorphAnalyzer
import fasttext
model = fasttext.load_model('lid.176.ftz')

data_temp_ru=pickle.load(open('catalog_vec_tfidf_ru.pkl','rb'))
v_ru=pickle.load(open('model_tfidf_ru.pkl','rb'))
df_ru=pickle.load(open('catalog_ru.pkl','rb'))

data_temp_kk=pickle.load(open('catalog_vec_tfidf_kk.pkl','rb'))
v_kk=pickle.load(open('model_tfidf_kk.pkl','rb'))
df_kk=pickle.load(open('catalog_kk.pkl','rb'))

data_temp_en=pickle.load(open('catalog_vec_tfidf_en.pkl','rb'))
v_en=pickle.load(open('model_tfidf_en.pkl','rb'))
df_en=pickle.load(open('catalog_en.pkl','rb'))

patterns_ru = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
patterns_kk = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
patterns_en = "[А-Яа-я0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
#stopwords_ru = stopwords.words("russian")
stopwords_ru=['другои', 'еи', 'какои', 'мои', 'неи', 'сеичас', 'такои', 'этои','и', 'в', 'во', 'не', 'что', 'он','на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне','было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между']
stopwords_en=['i','me', 'my', 'myself', 'we', 'our', 'ours','ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any','both', 'each', 'few', 'more', 'most', 'other','some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stopwords_kk=['ах', 'ох', 'эх', 'ай', 'эй', 'ой', 'тағы', 'тағыда', 'әрине', 'жоқ', 'сондай', 'осындай', 'осылай', 'солай', 'мұндай', 'бұндай', 'мен', 'сен', 'ол', 'біз', 'біздер', 'олар', 'сіз', 'сіздер', 'маған', 'оған', 'саған', 'біздің', 'сіздің', 'оның', 'бізге', 'сізге','оларға', 'біздерге', 'сіздерге', 'оларға', 'менімен', 'сенімен', 'онымен', 'бізбен', 'сізбен', 'олармен', 'біздермен', 'сіздермен', 'менің', 'сенің', 'біздің', 'сіздің', 'оның', 'біздердің', 'сіздердің', 'олардың', 'маған', 'саған', 'оған', 'менен', 'сенен', 'одан', 'бізден', 'сізден', 'олардан', 'біздерден', 'сіздерден', 'олардан', 'айтпақшы', 'сонымен', 'сондықтан', 'бұл', 'осы', 'сол', 'анау', 'мынау', 'сонау', 'осынау', 'ана', 'мына', 'сона', 'әні', 'міне', 'өй', 'үйт', 'бүйт', 'біреу', 'кейбіреу', 'кейбір', 'қайсыбір', 'әрбір', 'бірнеше', 'бірдеме', 'бірнеше', 'әркім', 'әрне', 'әрқайсы', 'әрқалай', 'әлдекім', 'әлдене', 'әлдеқайдан', 'әлденеше', 'әлдеқалай', 'әлдеқашан', 'алдақашан', 'еш', 'ешкім', 'ешбір', 'ештеме', 'дәнеңе', 'ешқашан', 'ешқандай', 'ешқайсы', 'емес', 'бәрі', 'барлық', 'барша', 'бар', 'күллі', 'бүкіл', 'түгел', 'өз', 'өзім', 'өзің', 'өзінің', 'өзіме', 'өзіне', 'өзімнің', 'өзі', 'өзге', 'менде', 'сенде', 'онда', 'менен', 'сенен\tонан', 'одан', 'ау', 'па', 'ей', 'әй', 'е', 'уа', 'уау', 'уай', 'я', 'пай', 'ә', 'о', 'оһо', 'ой', 'ие', 'аһа', 'ау', 'беу', 'мәссаған', 'бәрекелді', 'әттегенай', 'жаракімалла', 'масқарай', 'астапыралла', 'япырмай', 'ойпырмай', 'кәне', 'кәнеки', 'ал', 'әйда', 'кәні', 'міне', 'әні', 'сорап', 'қош-қош', 'пфша', 'пішә', 'құрау-құрау', 'шәйт', 'шек', 'моһ', 'тәк', 'құрау', 'құр', 'кә', 'кәһ', 'күшім', 'күшім', 'мышы', 'пырс', 'әукім', 'алақай', 'паһ-паһ', 'бәрекелді', 'ура', 'әттең', 'әттеген-ай', 'қап', 'түге', 'пішту', 'шіркін', 'алатау', 'пай-пай', 'үшін', 'сайын', 'сияқты', 'туралы', 'арқылы', 'бойы', 'бойымен', 'шамалы', 'шақты', 'қаралы', 'ғұрлы', 'ғұрлым', 'шейін', 'дейін', 'қарай', 'таман', 'салым', 'тарта', 'жуық', 'таяу', 'гөрі', 'бері', 'кейін', 'соң', 'бұрын', 'бетер', 'қатар', 'бірге', 'қоса', 'арс', 'гүрс', 'дүрс', 'қорс', 'тарс', 'тырс', 'ырс', 'барқ', 'борт', 'күрт', 'кірт', 'морт', 'сарт', 'шырт', 'дүңк', 'күңк', 'қыңқ', 'мыңқ', 'маңқ', 'саңқ', 'шаңқ', 'шіңк', 'сыңқ', 'таңқ', 'тыңқ', 'ыңқ', 'болп','былп', 'жалп', 'желп', 'қолп', 'ірк', 'ырқ', 'сарт-сұрт', 'тарс-тұрс', 'арс-ұрс', 'жалт-жалт', 'жалт-жұлт', 'қалт-қалт', 'қалт-құлт', 'қаңқ-қаңқ', 'қаңқ-құңқ', 'шаңқ-шаңқ', 'шаңқ-шұңқ', 'арбаң-арбаң', 'бүгжең-бүгжең', 'арсалаң-арсалаң', 'ербелең-ербелең', 'батыр-бұтыр', 'далаң-далаң','тарбаң-тарбаң', 'қызараң-қызараң', 'қаңғыр-күңгір', 'қайқаң-құйқаң', 'митың-митың', 'салаң-сұлаң', 'ыржың-тыржың', 'бірақ', 'алайда', 'дегенмен', 'әйтпесе', 'әйткенмен', 'себебі', 'өйткені', 'сондықтан', 'үшін', 'сайын', 'сияқты', 'туралы', 'арқылы', 'бойы', 'бойымен', 'шамалы', 'шақты', 'қаралы', 'ғұрлы', 'ғұрлым', 'гөрі', 'бері', 'кейін', 'соң', 'бұрын', 'бетер', 'қатар', 'бірге', 'қоса', 'шейін', 'дейін', 'қарай', 'таман', 'салым', 'тарта', 'жуық', 'таяу', 'арнайы', 'осындай', 'ғана', 'қана', 'тек', 'әншейін']

morph = MorphAnalyzer()    

app = Flask(__name__)

def result_v1(search_text,data_temp=None,v=None,df=None,patterns=None,stopwords=None,morph=None):

    
    #from sklearn.metrics.pairwise import cosine_similarity
    from scipy.spatial.distance import cosine
    import numpy as np
    import pandas as pd
    import re
    
    


    def lemmatize(doc,patterns,stopwords):
        doc = re.sub(patterns, ' ', doc)
        tokens = []
        for token in doc.split():
            if token and token not in stopwords:
                token = token.strip()
                token = morph.normal_forms(token)[0]

                tokens.append(token.lower())
        if len(tokens) > 0:
            return tokens
        return None


    a=[search_text]
    try:
        a=[' '.join(pd.Series(a).apply(lambda x: lemmatize(x,patterns,stopwords))[0])]
        
    except Exception:
        a=['null']
        pass
    
    #print(a)
    a_vec=np.array(v.transform(a).T.todense()).flatten()
    t=[1-cosine(a_vec, np.array((data_temp[i]))) for i in range(data_temp.shape[1])]

#     from joblib import Parallel, delayed
#     def process(i):
#         return cosine(a_vec,i)
#     t = Parallel(n_jobs=2)(delayed(process)(i) for i in range(data_temp.shape[1]))


    t = np.array(t)
    
    
    inds = (-t).argsort()
    
    top=10
    code2=df['CODE'].loc[inds[:top]].tolist()
    name2=df['NAME'].loc[inds[:top]].tolist()
    score2=t[inds[:top]].tolist()
    
    cutoff=np.where(np.array(score2)>0.0001,True,False)
    
    code2=[val for is_good, val in zip(cutoff, code2) if is_good]
    name2=[val for is_good, val in zip(cutoff, name2) if is_good]
    score2=[val for is_good, val in zip(cutoff, score2) if is_good]

    
    
    
    #t2=pd.DataFrame([t,df['CODE'],df['NAME']]).T.rename(columns={0:'SCORE',1:'CODE',2:'NAME'}).sort_values('SCORE',ascending=False).head(10)[['CODE','NAME','SCORE']]
    #return df['CODE'].loc[inds[:10]].tolist(),df['NAME'].loc[inds[:10]].tolist(),t[inds[:10]].tolist()
    return code2,name2,score2
    
@app.route('/', methods=['POST','GET'])    
def my_form_post():
    
    start = time.time()

    
    a='пример' 

    #newList=[{'name': 'red', 'code': 'a1', 'score': 1}, {'name': 'green', 'code': 'a5', 'score': 2}, {'name': 'blue', 'code': 'a11', 'score': 3}, {'name': 'blue2', 'code': 'a2', 'score': 5}]
    
    if request.method=="POST":
        a=request.form.get('name2')
        
    predict_language=model.predict(a, k=20)
    predict_language_lan,_=predict_language
    for i in predict_language_lan:
        if i in ['__label__en','__label__ru','__label__kk']:
            result=i[-2:]
            break
    
    if result=='ru':
        code,name,score=result_v1(a,data_temp_ru,v_ru,df_ru,patterns_ru,stopwords_ru,morph)
    elif result=='kk':
        code,name,score=result_v1(a,data_temp_kk,v_kk,df_kk,patterns_kk,stopwords_kk,morph)
    elif result=='en':
        code,name,score=result_v1(a,data_temp_en,v_en,df_en,patterns_en,stopwords_en,morph)       
        
   
    keys=['name','code','score']
    values=[name,code,score]

    newList = []
    for i in range(len(values[0])):

        tmp = {}
        for j, key in enumerate(keys):
            tmp[key] = values[j][i]

        newList.append(tmp)
        
    end = time.time()

    return render_template('catalog_tfidf.html',data=newList,time_calc=round(end-start,3))        


if __name__=='__main__':
    app.run(host='0.0.0.0', port=86, debug=False)