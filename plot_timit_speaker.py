from visualize_timit import *

r_list = [0, 1, 2, 3]
n_list = [50, 100, 300, 500]
m_list = [3, 5, 7, 9]
#r_list = list(reversed([0, 1, 2, 3, 4, 5, 6]))
#n_list = list(reversed([25, 50, 100, 300]))
#m_list = list(reversed([3, 5, 7, 9]))
print r_list, n_list, m_list
'''
    count, y_tag, x_tag = build_dict(timit_mlf, token_mlf)
    remove_sil(x_tag)
    y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
    mat = build_matrix(y_tag, x_tag, count, thresh)
    #x_tag = reorder_tag(mat.T, x_tag, y_tag)
    y_tag = reorder_tag(mat, y_tag, x_tag)
    y, x, y_range, x_range = plot_dict(y_tag, x_tag, count, thresh)
'''
def plot_block_debug(r, n, m, dot, thresh):
    timit_mlf = MLF('timit_train_{}_nosa.mlf'.format(dot))
    mlf_path = '/home/c2tao/timit_train_matdnn_nosa/tokenizer_bnf0_mr{}/{}_{}/result/result.mlf'
    token_mlf = MLF(mlf_path.format(r, n, m))
    assert(timit_mlf.wav_list==token_mlf.wav_list)
    
    count, y_tag, x_tag = build_dict(timit_mlf, token_mlf)
    remove_sil(x_tag)
    y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
    mat = build_matrix(y_tag, x_tag, count, thresh)

    #x_tag = reorder_tag(mat.T, x_tag, y_tag)
    #mat = build_matrix(y_tag, x_tag, count, thresh)
    y_tag = reorder_tag(mat, y_tag, x_tag)

    y, x, y_range, x_range = plot_dict(y_tag, x_tag, count, thresh)
    return count, y_tag, x_tag, np.array(y), np.array(x), np.array(y_range), np.array(x_range)
def plot_block(r, n, m, dot, thresh, x_tag_ext=None, rand=False):
    timit_mlf = MLF('timit_train_{}_nosa.mlf'.format(dot))
    mlf_path = '/home/c2tao/timit_train_matdnn_nosa/tokenizer_bnf0_mr{}/{}_{}/result/result.mlf'
    token_mlf = MLF(mlf_path.format(r, n, m))
    assert(timit_mlf.wav_list==token_mlf.wav_list)
    
    count, y_tag, x_tag = build_dict(timit_mlf, token_mlf)
    remove_sil(x_tag)

    '''
    y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
    mat = build_matrix(y_tag, x_tag, count, thresh)
    if not x_tag_ext:
        x_tag = reorder_tag(mat.T, x_tag, y_tag)
    else:
        x_tag = x_tag_ext 
    if not rand: 
        mat = build_matrix(y_tag, x_tag, count, thresh)
    y_tag = reorder_tag(mat, y_tag, x_tag)
    '''

    y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
    if not x_tag_ext:
        mat = build_matrix(y_tag, x_tag, count, thresh)
        x_tag = reorder_tag(mat.T, x_tag, y_tag)
    else:
        x_tag = x_tag_ext 
    y_tag, ___ = filter_dict(y_tag, x_tag, count, thresh)
    mat = build_matrix(y_tag, x_tag, count, thresh)
    y_tag = reorder_tag(mat, y_tag, x_tag)
    
    '''
    if not x_tag_ext:
        y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
        mat = build_matrix(y_tag, x_tag, count, thresh)
        x_tag = reorder_tag(mat.T, x_tag, y_tag)
    else:
        y_tag, x_tag = filter_dict(y_tag, x_tag, count, thresh)
        x_tag = x_tag_ext 
    if not rand: 
        mat = build_matrix(y_tag, x_tag, count, thresh)
    y_tag = reorder_tag(mat, y_tag, x_tag)
    '''
    y, x, y_range, x_range = plot_dict(y_tag, x_tag, count, thresh)
    return count, y_tag, x_tag, np.array(y), np.array(x), np.array(y_range), np.array(x_range)


#count, y_tag, x_tag, y, x, y_range, x_range = plot_block(5, 300, 9, 'phn', 50) # good z vs s
#for i, r in enumerate(r_list): 
#    count, yn_tag, xn_tag, yn, xn, yn_range, xn_range = plot_block(r, 300, 9, 'phn', 50, x_tag) 

#count, y_tag, x_tag, y, x, y_range, x_range = plot_block(5, 100, 9, 'phn', 50) # good z vs s
#for i, n in enumerate(n_list): 
#    count, yn_tag, xn_tag, yn, xn, yn_range, xn_range = plot_block(1, n, 9, 'phn', 100, x_tag) 

#count, y_tag, x_tag, y, x, y_range, x_range = plot_block(1, 100, 3, 'wrd', 100) # good z vs s
#for i, m in enumerate(m_list): 
#    count, yn_tag, xn_tag, yn, xn, yn_range, xn_range = plot_block(1, 100, m, 'wrd', 100, x_tag) 

#good m
#count, y_tag, x_tag, y, x, y_range, x_range = plot_block(1, 100, 3, 'wrd', 500/3) 
#for i, m in enumerate(m_list): 
#    count, yn_tag, xn_tag, yn, xn, yn_range, xn_range = plot_block(1, 100, m, 'wrd', 500.0/3, x_tag) 

#count, y_tag, x_tag, y, x, y_range, x_range = plot_block(1, 100, 3, 'wrd', 500/3) 
#for i, n in enumerate(n_list): 
#    count, yn_tag, xn_tag, yn, xn, yn_range, xn_range = plot_block(1, n, 9, 'wrd', 1000/n**0.5, x_tag) 
#    if i!=0:
#        y = np.hstack([y, yn+i]) 
#        x = np.hstack([x, xn])
#    else:
#        y = yn
#        x = xn
def plot_vert(r, n, m, dot, thresh, x_tag_ext=None, reverse = False):
    if not r: v_list = r_list
    elif not n: v_list = n_list
    elif not m: v_list = m_list
    if reverse:
        v_list = v_list[::-1]
    for i, v in enumerate(v_list):
        if i==0: 
            if not x_tag_ext: 
                x_tag = None
            else:
                x_tag = x_tag_ext
        if not r: 
            vr, vn, vm = v, n, m  
        elif not n:
            vr, vn, vm = r, v, m  
        elif not m:
            vr, vn, vm = r, n, v
        count, yn_tag, xn_tag, yn, xn, yn_range, xn_range = plot_block(vr, vn, vm, dot, thresh, x_tag) 
        plt.gca().text(-0.1, 0.5-i, 'MR={}\nn={}\nm={}'.format(vr, vn, vm), horizontalalignment='left', verticalalignment='center')
        plt.gca().text(-0.145, 0.5-i, '('+chr(97+i)+')', horizontalalignment='left', verticalalignment='center',fontsize=16)
        plt.hlines(-i, 0, 1)

        if i!=0:
            y = np.hstack([y, yn-i]) 
            x = np.hstack([x, xn])
        else:
            y = yn
            x = xn
            x_tag = xn_tag
            x_range = xn_range

    plt.plot(x, y, '.')
    plt.xticks(x_range, x_tag, rotation='vertical')
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.tight_layout()
    plt.gcf().subplots_adjust(left=0.15)
    plt.show()
    return x_tag
def plot_one(r, n, m, dot, thresh, x_tag_ext=None, rand=False):
    count, y_tag, x_tag, y, x, y_range, x_range = plot_block(r, n, m, dot, thresh, x_tag_ext, rand) 
    #count, y_tag, x_tag, y, x, y_range, x_range = plot_block_debug(r, n, m, dot, thresh) 
    y_n = len(y_tag)
    plt.plot(x, y, '.')
    plt.xticks(x_range, x_tag, rotation=90)

    y_pos = np.concatenate((y_range[::10], [y_range[-1]+1.0/(y_n+1)]))[1:]
    y_tag = range(10,y_n,10) + [y_n]
    plt.yticks(y_pos, y_tag)
    plt.tight_layout()
    plt.show()
    return x_tag



def vector_occr(word_list, vocab_list):
    return np.array(map(lambda x: word_list.count(x), vocab_list),dtype=np.float32)

def matrix_from_mlf(mlf_path):
    mlf = MLF(mlf_path.format(r, n, m))
    speaker_list, vector_list = [], []
    vocab_list = mlf.tok_list
    for i, tags in enumerate(mlf.tag_list):
        speaker = '_'.join(mlf.wav_list[i].split('_')[1:3])
        vector = vector_occr(tags, vocab_list)
        if i!=0 and speaker_list[-1]==speaker:
            vector_list[-1]+=vector
        else:
            #if i!=0: print speaker_list[-1], vector_list[-1]
            speaker_list.append(speaker)
            vector_list.append(vector)
    matrix = np.array(vector_list)
    return matrix, speaker_list, vocab_list

def matrix_accu_attr(mat, speaker_list, attr):
    if attr=='gender':
        idx = 4
    elif attr=='region':
        idx = 2
    attr_list = sorted(list(set(map(lambda sp: sp[idx], speaker_list))))
    attr_idx = {attr: attr_list.index(attr) for attr in attr_list}
    matrix = np.zeros((len(attr_list), mat.shape[1]), dtype=np.float32)
    for i, sp in enumerate(speaker_list):
        j = attr_idx[sp[idx]]
        matrix[j, :] +=mat[i,:]
    return matrix, attr_list
def matrix_thresh(mat, thresh = None):
    if not thresh:
        pass
     
    

def speaker_phone_count(r, n, m, dot, thresh, x_tag_ext=None, rand=False):
    timit_mlf = MLF('timit_train_{}_nosa.mlf'.format(dot))
    mlf_path = '/home/c2tao/timit_train_matdnn_nosa/tokenizer_bnf0_mr{}/{}_{}/result/result.mlf'
    token_mlf = MLF(mlf_path.format(r, n, m))
    assert(timit_mlf.wav_list==token_mlf.wav_list)
    
    count, y_tag, x_tag = build_dict(timit_mlf, token_mlf)
    remove_sil(x_tag)

if __name__=='__main__':
    #count, y_tag, x_tag_phn, y, x, y_range, x_range = plot_block(1, 100, 3, 'wrd', 150) 
    #count, y_tag, x_tag_wrd, y, x, y_range, x_range = plot_block(1, 300, 9, 'phn', 100)
    #count, y_tag, x_tag_mr, y, x, y_range, x_range = plot_block(5, 300, 9, 'phn', 50) 
    #plot_vert(1, 100, None, 'wrd', 150, x_tag_phn)
    #plot_vert(1, 300, None, 'phn', 100, x_tag_wrd)
    #plot_vert(None, 300, 9, 'phn', 30, x_tag_mr)
 
    #plot_vert(None, 300, 9, 'phn', 50) 
    #plot_vert(1, None, 9, 'phn', 100) 
    #plot_vert(1, 100, None, 'wrd', 160) 

    #plot_one(5, 300, 9, 'phn', 50, None, True) 
    #plot_one(5, 300, 9, 'wrd', 20, None, False) ???

    ##### resubmission plots from here
    # candidates

    #ws = ['become','academic','another','brother','please','trouble','available','shortage']
    #ws = ['was','is','as','his','have','has','she']
    #plot_one(1, 100, 7, 'wrd', 4, ws, False) 
    #ph = plot_one(1, 100, 7, 'phn', 60, None, False) 
    #plot_vert(None, 100, 7, 'phn', 60, ph) 
    #plot_vert(1, None, 7, 'phn', 60, ph) 
    ####plot_vert(1, 100, None, 'phn', 60, ph) 
    #plot_vert(1, 100, None, 'wrd', 40, reverse = True) 
    #plot_vert(1, 100, None, 'wrd', 4, ws) 

    #th = 100
    #ws = plot_one(1, 100, 5, 'wrd', 4, ws) 
    #ph = plot_one(1, 100, 5, 'phn', th, None) 
    #plot_vert(None, 100, 5, 'phn', th, ph) 
    #plot_vert(1, None, 5, 'phn', th, ph) 
    #plot_vert(1, 100, None, 'wrd', 4, ws)
 
    ##### resubmission revision plots from here
    r,m,n = 0, 5, 100
    mlf_path = '/home/c2tao/timit_train_matdnn_nosa/tokenizer_bnf0_mr{}/{}_{}/result/result.mlf'
    ''' 
    mlf = MLF(mlf_path.format(r, n, m))
    speaker_list, vector_list = [], []
        
    for i, tags in enumerate(mlf.tag_list):
        speaker = '_'.join(mlf.wav_list[i].split('_')[1:3])
        vector = vector_occr(tags, mlf.tok_list)
        if i!=0 and speaker_list[-1]==speaker:
            vector_list[-1]+=vector
        else:
            if i!=0: print speaker_list[-1], vector_list[-1]
            speaker_list.append(speaker)
            vector_list.append(vector)
    print speaker_list
    SV = np.array(vector_list)
    '''

    SV, s_list, v_list = matrix_from_mlf(mlf_path)
    print matrix_accu_attr(SV, s_list, 'gender')
    print matrix_accu_attr(SV, s_list, 'region')
    plt.matshow(np.log(SV+1))
    plt.show()
    #print vector_list
    #for tok in token_mlf.tok_list:
    #    map(lambda x: token_mlf.tag_list[0].count(x), )


