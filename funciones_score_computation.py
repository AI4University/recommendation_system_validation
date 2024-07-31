import pandas as pd
import numpy as np

def get_score_similarity(ranking, id_researcher, call):
    '''
    Function for obtaining the score based on the similarity of a recommendation

    ranking -> ranking of recommendations 
    id_researcher -> target researcher
    call -> target call
                score_similarity = 1 - ((max_similarity - similarity)/max_similarity)    

    '''
    max_similarity = ranking['similarity'][0]

    # detect if we are recommending researchers or calls
    if max_similarity != 0:
        if 'id_researcher' in ranking.columns.tolist():
            similarity = ranking[ranking['id_researcher']==id_researcher].reset_index()['similarity'][0]
        else:
            similarity = ranking[ranking['Call']==call].reset_index()['similarity'][0]

        return 1 - ((max_similarity - similarity)/max_similarity)
    else:
        return 0
    
def get_score_position(ranking, id_researcher, call):
    '''
    Function for obtaining the score based on the position of a recommendation in the ranking

    ranking -> ranking of recommendations 
    id_researcher -> target researcher
    call -> target call
    '''
    total_recommendations = ranking.shape[0]
    
    # detect if we are recommending researchers or calls
    if 'id_researcher' in ranking.columns.tolist():
        similarity = ranking[ranking['id_researcher']==id_researcher].reset_index()['similarity'][0]
        indice_valor_exacto = ranking.loc[ranking['id_researcher'] == id_researcher].index[0]
        ranking = ranking.iloc[:indice_valor_exacto + 1]

    else:
        similarity = ranking[ranking['Call']==call].reset_index()['similarity'][0]
        indice_valor_exacto = ranking.loc[ranking['Call'] == call].index[0]
        ranking = ranking.iloc[:indice_valor_exacto + 1]

    #total_recommendations = ranking.shape[0]
    posicion = ranking.shape[0]

    if similarity == 0: 
        return 0 
    else:
        return 1 - ((posicion-1) / total_recommendations)
    


def get_score_cluster(ranking, id_researcher, call):
    '''
    Function for obtaining the score based on the percentage of clusters or departments in the recommendations matched with the target

    ranking -> ranking of recommendations 
    id_researcher -> target researcher
    call -> target call
    '''

    # detect if we are recommending researchers or calls
    if 'id_researcher' in ranking.columns.tolist():
        # if similarity is 0 return 0
        if ranking[ranking['id_researcher'] == id_researcher].reset_index()['similarity'][0] == 0.0:
            return 0

        indice_valor_exacto = ranking.loc[ranking['id_researcher'] == id_researcher].index[0]
        ranking = ranking.iloc[:indice_valor_exacto + 1]
        posicion = ranking.shape[0]

        department_correcto = ranking['Department'][posicion-1]
        count_department_correctos = 0
        for i in range(ranking.shape[0]):
            if ranking['Department'][i] == department_correcto:
                count_department_correctos += 1

        return count_department_correctos/ranking.shape[0] 


    else:
        if ranking[ranking['Call'] == call].reset_index()['similarity'][0] == 0.0:
            return 0

        indice_valor_exacto = ranking.loc[ranking['Call'] == call].index[0]
        ranking = ranking.iloc[:indice_valor_exacto + 1]
        posicion = ranking.shape[0]

        cluster_correcto = ranking['Work Programme'][posicion-1]
        count_cluster_correctos = 0
        for i in range(ranking.shape[0]):
            if ranking['Work Programme'][i] == cluster_correcto:
                count_cluster_correctos += 1

        return count_cluster_correctos/ranking.shape[0]

        



