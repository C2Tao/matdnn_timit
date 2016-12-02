#import matplotlib.pyplot as plt
from zrst.util import MLF


r_list = [0, 1, 2, 3, 4, 5, 6]
n_list = [25, 50, 100, 300]
m_list = [3, 5, 7, 9]

mlf_path = '/home/c2tao/ZRC_timit/tokenizer_bnf0_mr{}/{}_{}/result/result.mlf'

r = 0
n = 25
m = 9
def fetch(self, med_list):
    return_list = []
    for I, T, Q in zip(self.int_list, self.tag_list, med_list):
        R = []
        pos = 0
        pi = 0
        for i, t, j in zip(I, T, range(len(T))):
            if pos >= len(Q): break
            match = (pi <= Q[pos] and i > Q[pos])
            while match:
                pos += 1
                R += j,
                # R += t,
                if pos >= len(Q): break
                match = (pi <= Q[pos] < i)
            pi = i
        return_list += R,
        try:
            assert len(R) == len(Q)
        except:
            print I
            print T
            print Q
            print 
            ssfsdf
    assert len(return_list) == len(med_list)
    return return_list

token_mlf = MLF(mlf_path.format(r, n, m))
timit_mlf = MLF('timit_train_phn.mlf')
assert(timit_mlf.wav_list==token_mlf.wav_list)
for i, w in enumerate(timit_mlf.wav_list):
    try:
        assert timit_mlf.int_list[i][-1] == token_mlf.int_list[i][-1]
    except:
        print timit_mlf.int_list[i]
        print token_mlf.int_list[i]
        print i, timit_mlf.wav_list[i]
        fdfsdf 
print fetch(timit_mlf, token_mlf.med_list)
print fetch(token_mlf, timit_mlf.med_list)
