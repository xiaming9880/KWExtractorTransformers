from keras import backend as K
#  recall count the number of true positives from all positives that should be 
# Args:
     #  y_pred lista de positivos predecidos
     #  y_true lista de todos los posibles valores predecidos
def tp(y_pred, y_true):
    sum_true_positives = 0
    for key_pred in y_pred: 
        for key_true in y_true:
            if key_pred.lower() == key_true.lower():
                sum_true_positives+=1
    return sum_true_positives
#  false negatives return the number of elements that should have been positives but didn't
def fn(y_pred, y_true):
    sum_false_negative = 0
    for key_true in y_true:
        isFound = False
        for key_pred in y_pred: 
            if key_pred.lower() == key_true.lower():
                isFound = True
                break
        if not isFound:
            sum_false_negative+=1
    return sum_false_negative

#  false positives return the number of elements that have been positives but they shoudn't
def fp(y_pred, y_true):
    sum_false_positives = 0
    for key_pred in y_pred: 
        isFound = False
        for key_true in y_true:
            if key_pred.lower() == key_true.lower():
                isFound = True
                break
        if not isFound:
            sum_false_positives+=1
    return sum_false_positives

#f1-score es la relacion entre precission and sensitive
# 2 (Precision*Sensitive)/(Precision-Sensitive) -> 2TP/(2TP+FP+FN)
def f1_score(tp,fp,fn):
    return 2*tp/(2*tp+fp+fn)

#precision is the fraction of the good responses divided by all rettrieved responses
def precision(tp,fp,fn):
    if (tp+fp)>0:
        return tp/(tp+fp)
    else:
        return 0

#recall is the fraction of the relevant documents that are sucesfully retrieved
def recall(tp,fp,fn):
    if (tp+fn)>0:
        return tp/(tp+fn)
    else:
        return 0

def main():
    keys=['scale', 'dimension', 'inertia', 'boundaries', 'positive power of ten', 'metre', 'kilogram', 'second', 'kelvin', 'ampere', 'candela', 'scientific notation', 'scales', 'measurement', 'matter', 'dimensions', 'space', 'movement', 'gravity', 'physical bodies', 'physical systems', 'morphology', 'macroscopic properties', 'microscopic properties', 'microscopic', 'direct observation', 'indirect observation', 'scales of observation', 'measurement', 'unit', 'quantitative properties', 'qualitative properties', 'instruments', 'base quantities', 'length', 'mass', 'time', 'temperature', 'electric current', 'luminous', 'intensity', 'amount of substance', 'volume', 'universal', 'international system of units']
    pred=['matter', 'classify',  'international', 'system', 'units', 'volume', 'liquid', 'mass', 'solid', 'ping', 'ball',  'gravity', 'physical', 'form', 'body', 'lorry', 'gravitation', 'classify matter', 'scientific notation', 'international system', 'physical bodies', 'physical systems', 'table', 'macro', 'microscopic', 'scale', 'density', 'observation', 'experimental', 'direct', 'tunnel', 'microscope', 'growth', 'phenomenon', 'plant', 'cell', 'division', 'indirect',   'model', 'diversity', 'size', 'study', 'nucleus', 'atom', 'diameter', 'universe', 'number', 'power', 'orders', 'magnitude', 'macroscopic', 'tunnelling', 'observable', 'physical system', 'observing matter', 'macroscopic scale', 'microscopic scale', 'direct observation', 'tunnelling microscope', 'macroscopic phenomenon', 'indirect observation', 'observable experimental', 'nuclear fusion model', 'direct indirect observation']
    print("True positive =",tp(pred,keys))
    print("False negative =",fn(pred,keys))
    print("False positive =",fp(pred,keys))
    print("F1 score =",f1_score(tp(pred,keys),fn(pred,keys),fp(pred,keys)))
   # -- quitar terminaciones -- ing añade muchos hijos que no hacen falta
   # quitar plurales e hijos que ya están siendo usados
   # añaidr of a las tuplas
#main()