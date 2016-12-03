#import matplotlib.pyplot as plt
from zrst.util import MLF
import numpy as np
import sys


#r = 0
#n = int(sys.argv[1])#300
#m = int(sys.argv[2])#9
#dot = sys.argv[3] #wrd
#thresh = int(sys.argv[4]) #100
#token_mlf = MLF(mlf_path.format(r, n, m))
#timit_mlf = MLF('timit_train_{}.mlf'.format(dot))
#assert(timit_mlf.wav_list==token_mlf.wav_list)
#return token_mlf, timit_mlf

def build_dict(src_mlf, ext_mlf):
    # map median of tokens in ext_mlf to tokens in src_mlf
    pos_list = src_mlf.fetch(ext_mlf.med_list)

    fet_list = []
    for p, s in zip(pos_list, src_mlf.tag_list):
        fet_list.append(map(lambda x: s[x], p))

    count = {}
    for s in src_mlf.tok_list:
        for e in ext_mlf.tok_list:
            count[(e, s)] = 0.0
    for t_list, f_list in zip(ext_mlf.tag_list, fet_list):
        for i, t in enumerate(t_list):
            count[(t, f_list[i])]+=1.0
    return count, sorted(ext_mlf.tok_list), sorted(src_mlf.tok_list)


import matplotlib.pylab as plt

def build_matrix(y_tag, x_tag, count, thresh = None):
    mat = np.zeros([len(y_tag), len(x_tag)],dtype=np.float32)
    for i, a in enumerate(y_tag):
        for j, b in enumerate(x_tag):
            if not thresh:
                mat[i, j] =  count[(a, b)] 
            elif count[(a, b)] > thresh: 
                mat[i, j] =  1.0 
    return mat

def filter_dict(y_tag, x_tag, count, thresh = 100):
    xy_lab = []
    for i, a in enumerate(y_tag):
        for j, b in enumerate(x_tag):
            if count[(a, b)]> thresh:
                xy_lab.append((b, a))
    x_lab, y_lab = zip(*xy_lab)
    x_tag = sorted(list(set(x_lab)))
    y_tag = sorted(list(set(y_lab)))
    #x_tag = list(set(x_lab))
    #y_tag = list(set(y_lab))
    return y_tag, x_tag

def filter_dict_top(y_tag, x_tag, count, thresh = 100, top = 5):
    mat = build_matrix(y_tag, x_tag, count, thresh)
    y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
    xy_lab = []
    for i in range(len(x_tag)):
        top_y_tag = np.argsort(mat[:, i])[:top]
        for j in top_y_tag:
            xy_lab.append((x_tag[i], y_tag[j]))
    x_lab, y_lab = zip(*xy_lab)
    x_tag = sorted(list(set(x_lab)))
    y_tag = sorted(list(set(y_lab)))
    return y_tag, x_tag

def plot_dict(y_tag, x_tag, count, thresh = 100):
    x_range = np.array(range(len(x_tag)), dtype=np.float32)/len(x_tag)
    y_range = np.array(range(len(y_tag)), dtype=np.float32)/len(y_tag)
    xy_list = []
    xy_lab = []
    for i, a in enumerate(y_tag):
        for j, b in enumerate(x_tag):
            if count[(a, b)]> thresh:
                xy_list.append((x_range[j], y_range[i]))
                xy_lab.append((b, a))
    x, y = zip(*xy_list)
    x_lab, y_lab = zip(*xy_lab)
    return y, x, y_range, x_range


def reorder_center(n):
    order = []
    if n%2==0:
        m = n/2
        for i in range(m):
            order.append(m-i-1)
            order.append(m+i)
    else:
        m = (n-1)/2
        order.append(m)
        for i in range(m):
            order.append(m-i-1)
            order.append(m+i+1)
    assert len(order)==n
    return np.array(order)


def reorder_tag(mat, y_tag, x_tag):
    #given x_tag order, sort y_tag
    new_y_order = []
    x_order = reorder_center(len(x_tag))
    y_order = reorder_center(len(y_tag))
    cy = 0 if len(y_tag)%2==0 else 1
    cx = 0 if len(x_tag)%2==0 else 1
    #print x_tag
    #print x_order
    #print y_tag
    #print y_order
    for e, o in enumerate(x_order):
        #print np.where(mat[:, o])[0]
        for y in np.where(mat[:, o])[0]:
            if y not in new_y_order:
                if e%2==cx:
                    new_y_order.append(y)
                else:
                    new_y_order.insert(0, y)
    #print len(new_y_order)
    #print len(y_order)
    
    # not sure why some values are missing
    for y in y_order:
        if y not in new_y_order:
            new_y_order.append(y)
    
    new_y_order = np.array(new_y_order)
    assert len(new_y_order)==len(y_order)

    new_y_tag = []
    for e, i in enumerate(new_y_order[y_order]):
        if e%2==cy:
            new_y_tag.append(y_tag[i])
        else:
            new_y_tag.insert(0, y_tag[i])
            
    return new_y_tag
     
def remove_sil(tag):
    rm = ['sil', 'h#']
    for r in rm:
        if r in tag:
            tag.remove(r)

def reorder_matrix(X, y_tag):
    from sklearn.decomposition import PCA
    #tags = ['0','1','2','3','4','5']
    #X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=1)
    order = pca.fit_transform(X)[:,0]
    #idx_order = np.argsort(order)
    return zip(*sorted(zip(y_tag, range(len(y_tag))), key=lambda (t, i): order[i]))[0]

def plot_fig2():
    r_list = [0, 1, 2, 3, 4, 5, 6]
    n_list = [25, 50, 100, 300]
    m_list = [3, 5, 7, 9]
    
    r, n, m, dot, thresh = 5, 300, 9, 'phn', 50
    #r, n, m, dot, thresh = 5, 300, 9, 'wrd', 100

    mlf_path = '/home/c2tao/ZRC_timit/tokenizer_bnf0_mr{}/{}_{}/result/result.mlf'
    token_mlf = MLF(mlf_path.format(r, n, m))
    timit_mlf = MLF('timit_train_{}.mlf'.format(dot))
    assert(timit_mlf.wav_list==token_mlf.wav_list)

    count, y_tag, x_tag = build_dict(timit_mlf, token_mlf)
    remove_sil(x_tag)
    y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
    mat = build_matrix(y_tag, x_tag, count, thresh)
    x_tag = reorder_tag(mat.T, x_tag, y_tag)
    y_tag = reorder_tag(mat, y_tag, x_tag)
    y, x, y_range, x_range = plot_dict(y_tag, x_tag, count, thresh)
    print len(x_tag), len(y_tag)

    plt.plot(x, y, '.')
    plt.xticks(x_range, x_tag, rotation='vertical')
    #plt.yticks(y_range, y_tag, rotation='horizontal')
    plt.ylabel('Different Tokens')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()

#def find_overlap_matrix(mat, x_tag, y_tag):
    
    

def plot_fig1():
    r_list = [0, 1, 2, 3, 4, 5, 6]
    n_list = [25, 50, 100, 300]
    m_list = [3, 5, 7, 9]
    
    #r, n, m, dot, thresh = 5, 300, 9, 'phn', 50
    r, n, m, dot, thresh = 5, 300, 9, 'wrd', 100

    mlf_path = '/home/c2tao/ZRC_timit/tokenizer_bnf0_mr{}/{}_{}/result/result.mlf'
    token_mlf = MLF(mlf_path.format(r, n, m))
    timit_mlf = MLF('timit_train_{}.mlf'.format(dot))
    assert(timit_mlf.wav_list==token_mlf.wav_list)

    count, y_tag, x_tag = build_dict(timit_mlf, token_mlf)
    remove_sil(x_tag)
    y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
    mat = build_matrix(y_tag, x_tag, count, thresh)
    
    x_tag = reorder_tag(mat.T, x_tag, y_tag)
    #mat = build_matrix(y_tag, x_tag, count, thresh)#ignore this to shuffle

    y_tag = reorder_tag(mat, y_tag, x_tag)
    mat = build_matrix(y_tag, x_tag, count, thresh)
    print len(x_tag), len(y_tag)

    y, x, y_range, x_range = plot_dict(y_tag, x_tag, count, thresh)

    plt.plot(x, y, '.')
    plt.xticks(x_range, x_tag, rotation='vertical')
    #plt.yticks(y_range, y_tag, rotation='horizontal')
    plt.ylabel('Different Tokens')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.show()
if __name__=='__main__':
    plot_fig1()
    plot_fig2()
