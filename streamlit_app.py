import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
import math
import geopy
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium
import random
from config import *

start = {'skob': 11, 'litke': 7}[dataset]
vlen = {'skob': 3, 'litke': 1}[dataset]

line_dist0 = 2

st.set_page_config(layout="wide")

def format_linenum(line_idx):
    bounds = st.session_state['bounds']
    page_idx = int(np.where(line_idx <= bounds)[0][0])
    if page_idx > 0:
        line_idx -= bounds[page_idx-1]
    return page_idx, line_idx-1

def update_entities(data):
    loc_seq = st.session_state['loc_seq']
    entities = st.session_state['entities']
    for part in data:
        for d in part:
            try:
                if d['Тип'] not in entities:
                    entities[d['Тип']] = {}
                for a in d['Упоминания']:
                    a['Номер строки'] = int(a['Номер строки'])
                    a['page_idx'],a['line_idx'] = format_linenum(a['Номер строки'])
                if d['Имя'] not in entities[d['Тип']]:
                    entities[d['Тип']][d['Имя']] = {'Описание': d['Описание'], 'Упоминания': d['Упоминания']}
                else:
                    entities[d['Тип']][d['Имя']]['Описание'] += '\n\n' + d['Описание']
                    entities[d['Тип']][d['Имя']]['Упоминания'] += d['Упоминания']
                if d['Тип'] == 'Место':
                    loc_seq += [{'Имя': d['Имя'], 'Номер строки': m['Номер строки']} for m in d['Упоминания']]
            except:
                pass
    loc_seq.sort(key = lambda x: x['Номер строки']) 
    
    for ent_type in entities:
        for ent_name in entities[ent_type]:
            entities[ent_type][ent_name]['Упоминания'].sort(key = lambda x: x['Номер строки'])
    
    types = list(entities.keys())
    types.sort(key = lambda x: 0 if x == 'Персона' else 1 if x == 'Место' else 2)
    entities = {k: entities[k] for k in types}
    
    if not entities:
        entities = {'Тип': {'Имя': {'Описание': 'Описание', 'Упоминания': [{'Роль':'Роль','Номер строки':1,'page_idx':1,'line_idx':1}]}}}
    
    st.session_state['entities'] = entities
    st.session_state['loc_seq'] = loc_seq

def typeOnChange():
    try:
        idx = list(st.session_state['entities'].keys()).index(st.session_state['boxType'])
    except:
        idx = 0
    st.session_state['mention'] = [idx, 0, 0]
    st.session_state['force'] = False

def nameOnChange():
    ent_type = list(st.session_state['entities'].keys())[st.session_state['mention'][0]]
    try:
        idx = list(st.session_state['entities'][ent_type].keys()).index(st.session_state['boxName'])
    except:
        idx = 0
    st.session_state['mention'] = [st.session_state['mention'][0], idx, 0]
    st.session_state['force'] = False
    
def caseOnChange():
    try:
        idx = st.session_state['values'].index(st.session_state['boxCase'])
    except:
        idx = 0
    st.session_state['mention'] = st.session_state['mention'][:2] + [idx]
    st.session_state['force'] = False
    
def prevOnClick():
    st.session_state['page_idx'] -= 1
    page_idx = st.session_state['page_idx']
    st.session_state['force'] = not sum(len(t) for t in st.session_state['pagewise'][page_idx])
    if not st.session_state['force']:
        ment = [t for t in st.session_state['pagewise'][page_idx] if len(t)][0][0]
        st.session_state['mention'] = ment[1]
    
def pageOnChange():
    fn = st.session_state['boxPage']
    page_idx = st.session_state['filenames'].index(fn)
    st.session_state['page_idx'] = page_idx
    st.session_state['force'] = not sum(len(t) for t in st.session_state['pagewise'][page_idx])
    if not st.session_state['force']:
        ment = [t for t in st.session_state['pagewise'][page_idx] if len(t)][0][0]
        st.session_state['mention'] = ment[1]

def nextOnClick():
    st.session_state['page_idx'] += 1
    page_idx = st.session_state['page_idx']
    st.session_state['force'] = not sum(len(t) for t in st.session_state['pagewise'][page_idx])
    if not st.session_state['force']:
        ment = [t for t in st.session_state['pagewise'][page_idx] if len(t)][0][0]
        st.session_state['mention'] = ment[1]
        
def update_distances():
    entities = st.session_state['entities']
    num_lines = sum([len([l for l in m['lines'] if l != '#']) for i,m in enumerate(st.session_state['markup']) \
                     if sum(st.session_state['pagewise'][i],[])])
    if 'sliderDist' not in st.session_state:
        line_dist = line_dist0
    else:
        line_dist = st.session_state['sliderDist']
    
    p = (2*line_dist+1) / max(1,num_lines)
    distances = np.zeros((len(st.session_state['line_idxs']), len(st.session_state['line_idxs'])), dtype=int)
    probs = np.zeros((len(st.session_state['line_idxs']), len(st.session_state['line_idxs'])))
    matches = {k: {} for k in st.session_state['line_idxs'].keys()}
    
    for i1,k1 in enumerate(st.session_state['line_idxs'].keys()):
        v1 = st.session_state['line_idxs'][k1]
        t1 = st.session_state['dates'][k1]
        for i2,k2 in enumerate(st.session_state['line_idxs'].keys()):
            v2 = st.session_state['line_idxs'][k2]
            t2 = st.session_state['dates'][k2]
            if i1 < i2:
                d = np.abs(v1[...,None] - v2) <= line_dist
                rows,cols = linear_sum_assignment(d, maximize=True)
                matches[k1][k2] = [(v1[a],v2[b],t1[a],t2[b]) for a,b in zip(rows,cols) if d[a,b]]
                matches[k2][k1] = [(v2[b],v1[a],t2[b],t1[a]) for a,b in zip(rows,cols) if d[a,b]]
                distances[i1,i2] = d[rows,cols].sum()
                distances[i2,i1] = distances[i1,i2]
                if len(matches[k1][k2]) > 0:
                    val = 0
                    k = min(len(v1),len(v2))
                    m = max(len(v1),len(v2))
                    q = 1 - (1-p) ** m
                    for n in range(len(matches[k1][k2]), k+1):
                        val += (q ** n) * ((1-q) ** (k-n)) * math.comb(k, n)                     
                    probs[i1,i2] = max(0, 1 - val)
                    probs[i2,i1] = max(0, 1 - val)
                
    st.session_state['matches'] = matches
    st.session_state['distances'] = pd.DataFrame(distances, st.session_state['line_idxs'].keys(), st.session_state['line_idxs'].keys())
    st.session_state['probs'] = pd.DataFrame(probs, st.session_state['line_idxs'].keys(), st.session_state['line_idxs'].keys())
        
def distOnChange():
    update_distances()
        
def make_pagewise():
    markup = st.session_state['markup']
    entities = st.session_state['entities']
    pagewise = [ [[] for _ in range(len(m['lines']))] for m in markup]
    st.session_state['line_max'] = 0
    st.session_state['line_idxs'] = {}
    st.session_state['dates'] = {}
    for jt,ent_type in enumerate(entities):
        entities[ent_type] = {k: v for k, v in sorted(list(entities[ent_type].items()), key=lambda x: x[0].lower())}
        for jn,ent_name in enumerate(entities[ent_type]):
            st.session_state['line_idxs'][f'{ent_name} ({ent_type})'] = np.array([m['Номер строки'] \
                                                                                  for m in entities[ent_type][ent_name]['Упоминания']])
            st.session_state['dates'][f'{ent_name} ({ent_type})'] = \
                list(map(lambda x: markup[x[0]]['dates'][x[1]], \
                [format_linenum(m['Номер строки']) for m in entities[ent_type][ent_name]['Упоминания']]))
            for jm,ent_ment in enumerate(entities[ent_type][ent_name]['Упоминания']):
                pagewise[ent_ment['page_idx']][ent_ment['line_idx']].append((ent_name,[jt,jn,jm]))
                st.session_state['line_max'] = max(st.session_state['line_max'], ent_ment['Номер строки'])
    st.session_state['entities'] = entities
    st.session_state['pagewise'] = pagewise
    update_distances()

def main():
        
    if 'geolocator' not in st.session_state:
        st.session_state['geolocator'] = Nominatim(user_agent=f"{dataset}-diary")
    if 'loc_seq' not in st.session_state:
        st.session_state['loc_seq'] = []
        
    if 'filenames' not in st.session_state:
        filenames = os.listdir('labels')
        filenames = [fn[:-len('.txt')] for fn in filenames if fn.endswith('.txt')]        
        
        filenames.sort(key=lambda x: 1000*int(x[start-2-vlen:start-2]) + int(''.join(c for c in x[start:] if c.isdigit())) + 0.25*('a' in x)+\
                       0.25*('об' in x))
        
        markup = []
        for fn in filenames:
            with open(f'labels/{fn}.txt', encoding='utf-8') as f:
                text = f.read().strip().split('\n')
                markup.append({'filename': fn,
                               'lines': text[1::3],
                               'boxes': [list(map(float, t.split())) for t in text[0::3]],
                               'dates': text[2::3],
                              })
        bounds = np.cumsum(np.array([len(m['lines']) for m in markup]))
        st.session_state['filenames'] = filenames
        st.session_state['markup'] = markup
        st.session_state['bounds'] = bounds
    else:
        filenames = st.session_state['filenames']
        markup = st.session_state['markup']
        bounds = st.session_state['bounds']
    
    if 'entities' not in st.session_state:
        files = os.listdir('jsons')
        files = sorted([fn for fn in files if fn.endswith('.json')])
        st.session_state['entities'] = {}
        data = []
        for fn in files:
            with open(f'jsons/{fn}', 'rb') as json_data:
                data.append(json.load(json_data))
        update_entities(data)
        make_pagewise()
    
    if 'mention' not in st.session_state:
        st.session_state['mention'] = next(t for t in next(s for s in st.session_state['pagewise'] if any(s)) if any(t))[0][1]
    if 'page_idx' not in st.session_state:
        st.session_state['page_idx'] = 0
    if 'force' not in st.session_state:
        st.session_state['force'] = False
    
    if 'locations' not in st.session_state:
        locs = pd.read_csv('locations.csv', encoding='utf-16', sep='\t')
        st.session_state['locations'] = {loc['Имя']: (loc['Широта'],loc['Долгота']) for i,loc in locs.iterrows()}

    entities = st.session_state['entities']
    pagewise = st.session_state['pagewise']
    locations = st.session_state['locations']
    
    ent_type = st.sidebar.selectbox('Выберите тип сущности:', entities.keys(), index=min(len(entities)-1,st.session_state['mention'][0]), \
                                    key='boxType', on_change=typeOnChange)
    ent_name = st.sidebar.selectbox('Выберите сущность:', entities[ent_type].keys(), \
                                    index=min(len(entities[ent_type])-1, st.session_state['mention'][1]), \
                                    key='boxName', on_change=nameOnChange)
    st.sidebar.text_area(label='Описание', value=entities[ent_type][ent_name]['Описание'], height=100, disabled=True, label_visibility="collapsed")
    mentions = entities[ent_type][ent_name]['Упоминания']
    values = list(map(lambda x: markup[x[0]]['dates'][x[1]] + ' – ' + x[2], \
                      [format_linenum(m['Номер строки'])+(m['Роль'],) for m in mentions]))
    #values = [(m['Дата'] if 'Дата' in m else 'Неизвестно') + ' – ' + m['Роль'] for m in mentions]
    st.session_state['values'] = values
    ent_case = st.sidebar.selectbox('Выберите упоминание:', values, index=min(len(values)-1, st.session_state['mention'][2]), \
                                    key='boxCase', on_change=caseOnChange)
    try:
        case_idx = values.index(ent_case)
    except:
        case_idx = 0
    
    if st.session_state['force']:
        page_idx,line_idx = st.session_state['page_idx'],-1
    else:
        page_idx,line_idx = mentions[case_idx]['page_idx'],mentions[case_idx]['line_idx']
    
    json_fn = st.sidebar.file_uploader('Загрузить свой список', type=['.json','.txt'], accept_multiple_files=True)
    append = st.sidebar.checkbox('Добавить к исходной базе', value=False, key='append_cbox')
    if 'json_names' not in st.session_state:
        st.session_state['json_names'] = []
    json_names = sorted([fn.name for fn in json_fn])
    
    if json_names != st.session_state['json_names']: 
        if json_names:
            if not append:
                st.session_state['entities'] = {}
            data = []
            for fn in json_fn:
                data.append(json.load(fn))
            update_entities(data)
        else:
            files = os.listdir('jsons')
            files = sorted([fn for fn in files if fn.endswith('.json')])
            st.session_state['entities'] = {}
            data = []
            for fn in files:
                with open(f'jsons/{fn}', 'rb') as json_data:
                    data.append(json.load(json_data))
            update_entities(data)
        make_pagewise()
        
        st.session_state['json_names'] = json_names
        st.session_state['mention'] = [0,0,0]
        st.session_state['force'] = False
        st.rerun()
    
    st.markdown("## Навигация по дневникам " + ("А.В. Сухово-Кобылина" if dataset=="skob" else "Ф.П. Литке"))
    st.session_state['page_idx'] = page_idx
    col1,col2 = st.columns([0.45,0.55])
    with col1:
        #st.markdown(f'##### {filenames[page_idx]}, строка {line_idx+1}')
        tab1,tab2,tab3 = st.tabs(['Сущности', 'Текст', 'Коллокации'])
        
        with tab1:
            if line_idx >= 0:
                step = 2
                text = ''.join('[' + str(i+1) + '] ' + l + '  \n' for i,l in zip(list(range(max(line_idx-step,0), line_idx)), \
                                                                                 markup[page_idx]['lines'][max(line_idx-step,0):line_idx]))
                text += '**[' + str(line_idx+1) + '] ' + markup[page_idx]['lines'][line_idx] + '**  \n'
                text += ''.join('[' + str(i+1) + '] ' + l + '  \n' for i,l in zip(list(range(line_idx+1, line_idx+step+1)), \
                                                                                  markup[page_idx]['lines'][line_idx+1:line_idx+step+1]))
                st.markdown(text.replace('#', '<...>').replace('.','\.'))

            for i,ments in enumerate(pagewise[page_idx]):
                lengths = [0*len(ment[0])+1 for ment in ments]
                cols = st.columns([0.12] + [0.88*l/sum(lengths) for l in lengths])
                with cols[0]:
                    st.button(str(i+1), key=f'{i}_{page_idx}', disabled=True)
                for j,ment,col in zip(list(range(len(ments))),ments,cols[1:]):
                    with col:
                        colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
                        color = colors[ment[1][0] % len(colors)]
                        color = 'rgb(' + (''.join( str( (int(color[i:i+2], 16)+255)//2)+',' for i in [1,3,5]))[:-1] + ')'
                        
                        mark = f'button-after-{i}-{j}'
                        st.markdown(
                            """
                            <style>
                            .element-container:has(style){
                                display: none;
                            }
                            #button-after {
                                display: none;
                            }
                            .element-container:has(#""" + mark + """) {
                                display: none;
                            }
                            .element-container:has(#""" + mark + """) + div button {
                                background-color: """ + color + """;
                                }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.markdown(f'<span id="{mark}"></span>', unsafe_allow_html=True)
                        if st.button(ment[0], key=f'btn_{i}_{j}'):
                            st.session_state['mention'] = ment[1]
                            st.session_state['force'] = False
                            st.rerun()
                            
        with tab2:
            text = ''.join('[' + str(i+1) + '] ' + l + '  \n' for i,l in enumerate(markup[page_idx]['lines']))
            st.text(text.replace('#', '<...>'))
            
        with tab3:
            tag = f'{ent_name} ({ent_type})'
            line_dist = st.slider('Длина контекста', 0, 10, line_dist0, key='sliderDist', on_change=distOnChange)
            try:
                probs = st.session_state['probs'][tag].sort_values(ascending=False)
                dists = st.session_state['distances'][tag]
            except:
                update_distances()
            dists = dists[probs.keys()]
            
            pair = st.selectbox('Дополнительная сущность: ', [f'{v}, {100*w:.2f}% – {k}' for k,v,w in zip(dists.keys(), dists.values, probs.values) if k != tag])
            if pair:
                tag1 = pair[pair.find('–')+2:]

                matches = st.session_state['matches'][tag][tag1]
                text = ''
                for row,col,date1,date2 in matches:
                    p1,l1 = format_linenum(row)
                    p2,l2 = format_linenum(col)
                    if date1 == date2:
                        text += '**' + date1 + '**  \n'
                    else:
                        if row > col:
                            date1,date2 = date2,date1
                        if date1[-9:] == date2[-9:]:
                            text += '**' + date1[:-9] + '-' + date2[:-9] + date1[-9:] + '**  \n'
                        elif date1[-5:] == date2[-5:]:
                            text += '**' + date1[:-5] + ' – ' + date2[:-5] + date1[-5:] + '**  \n'
                        else:
                            text += '**' + date1 + ' – ' + date2 + '**  \n'
                    text += '['+ filenames[p1][start-2-vlen:] + '/' + str(l1+1) + '] ' + markup[p1]['lines'][l1] + '  \n'
                    text += '['+ filenames[p2][start-2-vlen:] + '/' + str(l2+1) + '] ' + markup[p2]['lines'][l2] + '  \n'
                st.markdown(text.replace('#', '<...>').replace('.','\.'))

                line_idxs = st.session_state['line_idxs'][tag] - 1
                line_idxs1 = st.session_state['line_idxs'][tag1] - 1
                sigma = 10
                yt = np.arange(1, st.session_state['line_max']+1)
                xt = (1/np.sqrt(2*np.pi*sigma) * np.exp(-1/2*((yt-line_idxs[...,None]-1)/sigma)**2)).sum(axis=0)
                xt1 = (1/np.sqrt(2*np.pi*sigma) * np.exp(-1/2*((yt-line_idxs1[...,None]-1)/sigma)**2)).sum(axis=0)

                if True:
                    fig,ax = plt.subplots(figsize=(5,10))
                    plt.ylim([0, st.session_state['line_max']])
                    plt.xticks([])
                    plt.plot(xt, yt, 'C0')
                    plt.plot(xt1, yt, 'C2--')
                    plt.plot(xt[line_idxs], yt[line_idxs], 'C0o')
                    plt.plot(xt1[line_idxs1], yt[line_idxs1], 'C2o')

                    plt.plot(xt[[m[0]-1 for m in matches]], [m[0] for m in matches], 'C3o')
                    plt.plot(xt1[[m[1]-1 for m in matches]], [m[1] for m in matches], 'C4o')

                    plt.plot(xt[line_idxs[case_idx]], yt[line_idxs[case_idx]], 'C1o')
                    ax.invert_yaxis()
                    ax.set_ylabel('Номер строки', rotation=-90)
                    plt.title(f'{len(line_idxs)}/{len(line_idxs1)}')
                    ax.legend([tag[:tag.find('(')-1], tag1[:tag1.find('(')-1]])
                else:
                    fig,ax = plt.subplots(figsize=(10,5))
                    plt.xlim([0, st.session_state['line_max']])
                    plt.yticks([])
                    plt.plot(yt, xt, 'C0')
                    plt.plot(yt, xt1, 'C2--')
                    plt.plot(yt[line_idxs], xt[line_idxs], 'C0o')
                    plt.plot(yt[line_idxs1], xt1[line_idxs1], 'C2o')

                    plt.plot([m[0] for m in matches], xt[[m[0]-1 for m in matches]], 'C3o')
                    plt.plot([m[1] for m in matches], xt1[[m[1]-1 for m in matches]], 'C4o')
                    ax.set(xlabel='Номер строки')
                    plt.title(f'{len(line_idxs)}/{len(line_idxs1)}')
                    ax.legend([tag[:tag.find('(')-1], tag1[:tag1.find('(')-1]])

                st.pyplot(fig, use_container_width=False)
        
    with col2:
        
        tab1,tab2,tab3 = st.tabs(['Изображение','Карта','Показатели'])
        
        with tab1:
            _,col21,col22,col23,_ = st.columns([0.01,0.24,0.4,0.24,0.11])
            with col21:
                st.button('Предыдущая', key='btnPrev', use_container_width=True, disabled=page_idx==0, on_click=prevOnClick)
            with col22:
                st.selectbox('Выберите страницу', filenames, key='boxPage', label_visibility='collapsed', index=page_idx, \
                             on_change=pageOnChange)
            with col23:
                st.button('Следующая', key='btnNext', use_container_width=True, disabled=page_idx==len(filenames)-1, on_click=nextOnClick)

            img = cv2.imdecode(np.fromfile(f"images/{filenames[page_idx] + '.jpg'}", dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            fig,ax = plt.subplots(figsize=(5,5))
            ax.set_axis_off()
            ax.imshow(img[...,(2,1,0)])
            h,w = img.shape[0],img.shape[1]

            for i,box in enumerate(markup[page_idx]['boxes']):
                plt.plot(np.array(box)[[1,3,5,7,1]]*w, np.array(box)[[2,4,6,8,2]]*h, linewidth=0.5, color='r' if i == line_idx else 'b')
            st.pyplot(fig, use_container_width=False)
            
        with tab2:
            folium_map = folium.Map(attributionControl=0)
            
            for loc in locations:
                found = False
                for line in pagewise[page_idx]:
                    found = found or any(m[0] == loc for m in line)
                folium.Marker(location=st.session_state['locations'][loc], popup=loc, \
                              icon=folium.Icon(color='green' if found else 'red')).add_to(folium_map)
            loc_seq = st.session_state['loc_seq']
            
            for a,b in zip(loc_seq, loc_seq[1:]):
                if a['Имя'] in locations and b['Имя'] in locations and a['Имя'] != b['Имя']:
                    pg_idx, ln_idx = format_linenum(b['Номер строки'])
                    act = pg_idx == page_idx
                    color = 'green' if act else 'black'
                    folium.PolyLine(locations=[locations[a['Имя']], locations[b['Имя']]], color=color, weight=1).add_to(folium_map)
                    
                    ya = np.log(np.tan(np.pi/4 + (locations[a['Имя']][0]*np.pi/360))) * 180/np.pi
                    yb = np.log(np.tan(np.pi/4 + (locations[b['Имя']][0]*np.pi/360))) * 180/np.pi
                    xa = locations[a['Имя']][1]
                    xb = locations[b['Имя']][1]
                    yc = 0.45*ya + 0.55*yb
                    lon = (np.arctan(np.exp(yc * np.pi/180)) - np.pi/4) * 360/np.pi
                    lat = 0.45*xa + 0.55*xb
                    ang = np.arctan2(xb-xa, yb-ya) * 180/np.pi - 90
                    
                    text = '<font size="3"><b>[' + filenames[pg_idx][start-2-vlen:] + '/' + str(ln_idx+1) + '] ' + markup[pg_idx]['dates'][ln_idx] + \
                           '</b><br>' + markup[pg_idx]['lines'][ln_idx] + '</font>'
                    popup = folium.Popup(text, parse_html=False, max_width=300)
                    folium.RegularPolygonMarker(location=(lon,lat), color=color, fill_color=color, number_of_sides=3, radius=10, weight=1, \
                                                fill_opacity=1, popup=popup, rotation=ang).add_to(folium_map)
            st_folium(folium_map, use_container_width=False)
            
        with tab3:
            indicator = st.selectbox('Выберите показатель:', options=entities['Показатель'] if 'Показатель' in entities else None)
            if indicator:
                data = entities['Показатель'][indicator]['Упоминания']
                xt = [d['Номер строки'] for d in data]
                yt = [(lambda x: x if isinstance(x,float) or isinstance(x,int) else None)(d['Значение']) for d in data]
                
                text = ''
                for x in xt:
                    pg_idx, ln_idx = format_linenum(x)
                    text += '[' + str(x) + '] ' + markup[pg_idx]['lines'][ln_idx] + '  \n'
                xt,yt = zip(*((x,y) for x,y in zip(xt,yt) if y))
                
                fig,ax = plt.subplots(figsize=(10,6))
                plt.plot(xt, yt, 'C0-o')
                plt.grid(True)
                st.pyplot(fig, use_container_width=False)
                st.markdown(text.replace('#', '<...>').replace('.','\.'))    
    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass