import numpy as np
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class TextRankSummarizer:
    def __init__(self, damping_factor=0.85, max_iter=100, tol=1e-4):
        """
        TextRank文本摘要器
        
        Args:
            damping_factor: 阻尼因子，通常设为0.85
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.damping_factor = damping_factor
        self.max_iter = max_iter
        self.tol = tol
        
    def preprocess_text(self, text):
        """预处理文本，按逗号分句并清理"""
        # 使用逗号分句
        sentences = re.split(r'[。！？!?.,]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def extract_keywords(self, sentence):
        """提取句子中的关键词"""
        # 使用jieba进行词性标注
        words = pseg.cut(sentence)
        # 只保留名词、动词、形容词
        keywords = []
        for word, flag in words:
            if len(word) > 1 and flag.startswith(('n', 'v', 'a')):
                keywords.append(word)
        return ' '.join(keywords)
    
    def build_similarity_matrix(self, sentences):
        """构建句子相似度矩阵"""
        # 提取每个句子的关键词
        processed_sentences = [self.extract_keywords(sent) for sent in sentences]
        
        # 使用TF-IDF向量化
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            lowercase=False
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_sentences)
            # 计算余弦相似度
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        except ValueError:
            # 如果TF-IDF失败，使用简单的词汇重叠相似度
            similarity_matrix = self._simple_similarity_matrix(processed_sentences)
        
        return similarity_matrix
    
    def _simple_similarity_matrix(self, sentences):
        """简单的词汇重叠相似度计算"""
        n = len(sentences)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    words_i = set(sentences[i].split())
                    words_j = set(sentences[j].split())
                    if len(words_i | words_j) > 0:
                        similarity_matrix[i][j] = len(words_i & words_j) / len(words_i | words_j)
        
        return similarity_matrix
    
    def textrank(self, similarity_matrix):
        """TextRank算法计算句子权重"""
        n = len(similarity_matrix)
        # 初始化权重
        scores = np.ones(n) / n
        
        for _ in range(self.max_iter):
            prev_scores = scores.copy()
            
            for i in range(n):
                rank_sum = 0
                for j in range(n):
                    if i != j and similarity_matrix[j][i] > 0:
                        # 计算从j到i的权重传递
                        weight_sum = np.sum(similarity_matrix[j])
                        if weight_sum > 0:
                            rank_sum += similarity_matrix[j][i] / weight_sum * prev_scores[j]
                
                scores[i] = (1 - self.damping_factor) + self.damping_factor * rank_sum
            
            # 检查收敛
            if np.abs(scores - prev_scores).sum() < self.tol:
                break
        
        return scores
    
    def get_top_sentences_from_two_inputs(self, text1, text2, k=3):
        """
        从两个输入句子中获取各自前k个重要分句
        
        Args:
            text1: 第一个输入句子
            text2: 第二个输入句子
            k: 每个句子返回的分句数量
            
        Returns:
            dict: 包含两个句子各自前k个分句的字典
        """
        # 预处理两个句子，按逗号分句
        sentences1 = self.preprocess_text(text1)
        sentences2 = self.preprocess_text(text2)
        
        # 记录每个分句来源
        sentence_sources = []
        all_sentences = []
        
        # 标记来源并合并所有分句
        for i, sent in enumerate(sentences1):
            all_sentences.append(sent)
            sentence_sources.append(('text1', i))
            
        for i, sent in enumerate(sentences2):
            all_sentences.append(sent)
            sentence_sources.append(('text2', i))
        
        if len(all_sentences) == 0:
            return {'text1': [], 'text2': []}
        
        # 构建相似度矩阵（基于所有分句）
        similarity_matrix = self.build_similarity_matrix(all_sentences)
        
        # 运行TextRank算法获取所有分句的得分
        scores = self.textrank(similarity_matrix)
        
        # 创建包含得分、来源和索引的列表
        scored_sentences = []
        for i, (sent, score, source_info) in enumerate(zip(all_sentences, scores, sentence_sources)):
            scored_sentences.append({
                'sentence': sent,
                'score': score,
                'source': source_info[0],  # 'text1' 或 'text2'
                'original_index': source_info[1],  # 在原句子中的索引
                'global_index': i  # 在合并列表中的索引
            })
        
        # 按得分排序
        scored_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        # 分别获取两个句子的前k个分句
        text1_sentences = []
        text2_sentences = []
        
        for item in scored_sentences:
            if item['source'] == 'text1' and len(text1_sentences) < k:
                text1_sentences.append((item['sentence'], item['score'], item['original_index']))
            elif item['source'] == 'text2' and len(text2_sentences) < k:
                text2_sentences.append((item['sentence'], item['score'], item['original_index']))
            
            # 如果两个列表都已满，退出循环
            if len(text1_sentences) >= k and len(text2_sentences) >= k:
                break
        
        return {
            'text1': text1_sentences,
            'text2': text2_sentences
        }
    
    def summarize_two_texts(self, text1, text2, k=3, return_scores=False):
        """
        对两个输入句子生成摘要
        
        Args:
            text1: 第一个输入句子
            text2: 第二个输入句子
            k: 每个句子摘要的分句数量
            return_scores: 是否返回得分
            
        Returns:
            dict: 包含两个句子摘要的字典
        """
        top_sentences = self.get_top_sentences_from_two_inputs(text1, text2, k)
        
        if return_scores:
            return top_sentences
        else:
            # 按原文顺序排列并生成摘要
            result = {}
            
            # 处理text1的摘要
            if top_sentences['text1']:
                text1_sorted = sorted(top_sentences['text1'], key=lambda x: x[2])
                result['text1'] = '，'.join([sent[0] for sent in text1_sorted])
            else:
                result['text1'] = ""
            
            # 处理text2的摘要
            if top_sentences['text2']:
                text2_sorted = sorted(top_sentences['text2'], key=lambda x: x[2])
                result['text2'] = '，'.join([sent[0] for sent in text2_sorted])
            else:
                result['text2'] = ""
            
            return result
        
if __name__ == '__main__':

    trs = TextRankSummarizer()
    text1 = "原告人陈某甲诉称，时间时间时间凌晨，其与陈某丙、陈某乙在广海北门头吃宵夜，然后陈某丙叫他们一起去某花园找其女朋友，当他们达到某花园，其中有一个人过来叫他们离开，当他们准备离开时，还不到一分钟，被告人李某就带领一帮人拿着弹簧刀过来不问缘由就把他们捅伤，被告人捅了陈某甲两刀，导致其肝脏破裂，失血过多，休克，约过了15分钟左右他们醒过来，陈某乙见伤势严重立即开车将陈某甲送去医院，在他们离开时，被告人依然带人一直追他们，直至到达广海医院，陈某甲在广海医院进行多次抢救，因伤势严重，后转至台山市人民医院进行抢救，休息至今。 2 . 被告人的犯罪行为致使陈某甲经济及精神上均遭受巨大损失，具体损失为医疗费219539元、误工费33480元、护理费950元、住院伙食补助费1900元、营养费500元、交通费1000元等，共计597839元。 3 . 这些损失因被告人的故意伤害行为产生，依法应由被告人负责赔偿。 4 . 若陈某甲日后如因本次伤害的病情恶化所造成的损失，陈某甲保留继续追诉的权利。 5 . 请求判令被告人赔偿陈某甲医疗费、误工费、护理费、住院伙食补助费、营养费、交通费等损失共计597839元。"
    text2 = "被告人李某辩称其没有拿刀、石头打人，对公诉机关指控的其他事实没有意见。 2 . 认为量刑建议过重。 3 . 表示同意赔偿附带民事原告人陈某甲、陈某乙、陈某丙的损失。 4 . 被告人的辩护人兼附带民事诉讼代理人在庭审中提出被告人只打一个人，不应计三个人的伤，认为量刑过重。"

    t = trs.summarize_two_texts(text1, text2)
    print(t)