import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

def format_linenum(line_idx, bounds, filenames):
    page_idx = int(np.where(line_idx <= bounds)[0][0])
    if page_idx > 0:
        line_idx -= bounds[page_idx-1]
    return page_idx, line_idx-1

def update_entities(entities, bounds, filenames, data):
    for d in data:
        try:
            if d['Тип'] not in entities:
                entities[d['Тип']] = {}
            for a in d['Упоминания']:
                a['page_idx'],a['line_idx'] = format_linenum(a['Номер строки'], bounds, filenames)
            if d['Имя'] not in entities[d['Тип']]:
                entities[d['Тип']][d['Имя']] = {'Описание': d['Описание'], 'Упоминания': d['Упоминания']}
            else:
                entities[d['Тип']][d['Имя']]['Описание'] += ' ' + d['Описание']
                entities[d['Тип']][d['Имя']]['Упоминания'] += d['Упоминания']
        except:
            pass
    return entities

def typeOnChange():
    st.session_state['mention'] = [list(st.session_state['entities'].keys()).index(st.session_state['boxType']), 0, 0]
    st.session_state['force'] = False

def nameOnChange():
    ent_type = list(st.session_state['entities'].keys())[st.session_state['mention'][0]]
    st.session_state['mention'] = [st.session_state['mention'][0], \
                                   list(st.session_state['entities'][ent_type].keys()).index(st.session_state['boxName']), 0]
    st.session_state['force'] = False
    
def caseOnChange():
    st.session_state['mention'] = st.session_state['mention'][:2] + [st.session_state['values'].index(st.session_state['boxCase'])]
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
        
def main():
    
    if 'mention' not in st.session_state:
        st.session_state['mention'] = [0,0,0]
    if 'page_idx' not in st.session_state:
        st.session_state['page_idx'] = 0
    if 'force' not in st.session_state:
        st.session_state['force'] = False
    
    if 'filenames' not in st.session_state:
        filenames = os.listdir('labels')
        filenames = [fn.strip('.txt') for fn in filenames if fn.endswith('.txt')]
        filenames.sort(key=lambda x: 1000*int(x[6:8]) + (int(x[11:]) if x[11:].isdigit() else int(x[11:-2])) + 0.5*('об' in x))
        markup = []
        for fn in filenames:
            with open(f'labels/{fn}.txt', encoding='utf-8') as f:
                text = f.read().strip().split('\n')
                markup.append({'filename': fn, 'lines': text[1::3], 'boxes': [list(map(float, t.split())) for t in text[0::3]]})
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
        entities = {}
        for fn in files:
            with open(f'jsons/{fn}', 'rb') as json_data:
                data = json.load(json_data)
            entities = update_entities(entities, bounds, filenames, data)
        pagewise = [ [[] for _ in range(len(m['lines']))] for m in markup]
        for jt,ent_type in enumerate(entities):
            entities[ent_type] = {k: v for k, v in sorted(list(entities[ent_type].items()))}
            for jn,ent_name in enumerate(entities[ent_type]):
                for jm,ent_ment in enumerate(entities[ent_type][ent_name]['Упоминания']):
                    pagewise[ent_ment['page_idx']][ent_ment['line_idx']].append((ent_name,[jt,jn,jm]))
        st.session_state['entities'] = entities
        st.session_state['pagewise'] = pagewise
    else:
        entities = st.session_state['entities']
        pagewise = st.session_state['pagewise']
    
    ent_type = st.sidebar.selectbox('Выберите тип сущности:', entities.keys(), index=st.session_state['mention'][0], \
                                    key='boxType', on_change=typeOnChange)
    ent_name = st.sidebar.selectbox('Выберите сущность:', entities[ent_type].keys(), index=st.session_state['mention'][1], \
                                    key='boxName', on_change=nameOnChange)
    st.sidebar.text_area(label='Описание', value=entities[ent_type][ent_name]['Описание'], height=100, disabled=True, label_visibility="collapsed")
    mentions = entities[ent_type][ent_name]['Упоминания']
    values = [(('0' if m['Дата'][1] == ' ' else '') + m['Дата'] if 'Дата' in m else 'Неизвестно') + ' – ' + m['Роль'] for m in mentions]
    st.session_state['values'] = values
    ent_case = st.sidebar.selectbox('Выберите упоминание:', values, index=st.session_state['mention'][2], \
                                    key='boxCase', on_change=caseOnChange)
    ent_idx = values.index(ent_case)
    
    if st.session_state['force']:
        page_idx,line_idx = st.session_state['page_idx'],-1
    else:
        page_idx,line_idx = mentions[ent_idx]['page_idx'],mentions[ent_idx]['line_idx']
    
    json_fn = st.sidebar.file_uploader('Загрузить свой список', type=['.json','.txt'], accept_multiple_files=True)
    if 'json_names' not in st.session_state:
        st.session_state['json_names'] = []
        
    json_names = sorted([fn.name for fn in json_fn])
    if json_names != st.session_state['json_names']: 
        if json_names:
            entities = {}
            for fn in json_fn:
                data = json.load(fn)
                entities = update_entities(entities, bounds, filenames, data)
        else:
            files = os.listdir('jsons')
            files = sorted([fn for fn in files if fn.endswith('.json')])
            entities = {}
            for fn in files:
                with open(f'jsons/{fn}', 'rb') as json_data:
                    data = json.load(json_data)
                entities = update_entities(entities, bounds, filenames, data)
        pagewise = [ [[] for _ in range(len(m['lines']))] for m in markup]
        for jt,ent_type in enumerate(entities):
            entities[ent_type] = {k: v for k, v in sorted(list(entities[ent_type].items()))}
            for jn,ent_name in enumerate(entities[ent_type]):
                for jm,ent_ment in enumerate(entities[ent_type][ent_name]['Упоминания']):
                    pagewise[ent_ment['page_idx']][ent_ment['line_idx']].append((ent_name,[jt,jn,jm]))
        st.session_state['json_names'] = json_names
        st.session_state['entities'] = entities
        st.session_state['pagewise'] = pagewise
        st.session_state['mention'] = [0,0,0]
        st.session_state['force'] = True
        st.rerun()
    
    st.markdown("## Навигация по дневникам А.В. Сухово-Кобылина")
    st.session_state['page_idx'] = page_idx
    col1,col2 = st.columns([0.45,0.55])
    with col1:
        #st.markdown(f'##### {filenames[page_idx][:-4]}, строка {line_idx+1}')
        tab1,tab2 = st.tabs(['Сущности', 'Текст'])
        
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
                        if st.button(ment[0], key=f'btn_{i}_{j}'):
                            st.session_state['mention'] = ment[1]
                            st.session_state['force'] = False
                            st.rerun()
                            
        with tab2:
            text = ''.join('[' + str(i+1) + '] ' + l + '  \n' for i,l in enumerate(markup[page_idx]['lines']))
            st.markdown(text.replace('#', '<...>').replace('.','\.'))
        
    with col2:
        
        _,col21,col22,col23,_ = st.columns([0.01,0.24,0.4,0.24,0.11])
        with col21:
            st.button('Предыдущая', key='btnPrev', use_container_width=True, disabled=page_idx==0, on_click=prevOnClick)
        with col22:
            st.selectbox('Выберите страницу', filenames, key='boxPage', label_visibility='collapsed', index=page_idx, on_change=pageOnChange)
        with col23:
            st.button('Следующая', key='btnNext', use_container_width=True, disabled=page_idx==len(filenames)-1, on_click=nextOnClick)
        
        img = cv2.imdecode(np.fromfile(f"images/{filenames[page_idx] + '.jpg'}", dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_axis_off()
        ax.imshow(img[...,(2,1,0)])
        h,w = img.shape[0],img.shape[1]

        for i,box in enumerate(markup[page_idx]['boxes']):
            plt.plot(np.array(box)[[1,3,5,7,1]]*w, np.array(box)[[2,4,6,8,2]]*h, linewidth=0.5, color='r' if i == line_idx else 'b')
        st.pyplot(fig, use_container_width=False)
    
    
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass