import os
import sys
import time
import gensim
import pprint
import numpy as np
import tensorflow as tf
from utils import progress, LearningRateUpdater, INFO_LOG, LOAD_READER, LOAD_CONFIG_MODEL, LOAD_TF_CONFIG, load_embedding
from evaluate import compute_metrics, compute_metrics_all, ndcg_at_k

flags = tf.flags
flags.DEFINE_string(
    "data_path",
    "./",
    "Where the training/test data is stored.")
flags.DEFINE_string(
    "model_path",
    "./",
    "Where the training/test data is stored.")
flags.DEFINE_string("dataset", "ptb", "which dataset")
flags.DEFINE_string("model_name", "lstm", "which model")

flags.DEFINE_string(
    "embedding_path",
    "/data/word2vec/glove.840B.300d.w2v.bin",
    "")
flags.DEFINE_boolean("use_embeddings", False, "reload from saved model")
flags.DEFINE_boolean("update_embeddings", False, "reload from saved model")

flags.DEFINE_boolean("reload", False, "reload from saved model")
flags.DEFINE_boolean("is_test", False, "test or not")
FLAGS = flags.FLAGS


def load_input_feed(batch, model, state, batch_size):
    input_feed = {
        # model.xplen: batch['para_len'],
        model.xfirst_sents: batch['sents'][:, 0, :],
        model.xfirst_lens: batch['seq_lens'][:, 0],
        model.xlast_sents: batch['sents'][:, -1, :],
        model.xlast_lens: batch['seq_lens'][:, -1],
        model.initial_state: state}

    middles_sents = batch['sents'][:, 1:-1, :]
    middles_rsts = batch['rsts'][:, 1:-1, :]
    middles_lens = batch['seq_lens'][:, 1:-1]

    for i in range(model.para_len - 2):
        input_feed[model.xmiddles_sents[i].name] = middles_sents[:, i, :]
        input_feed[model.xmiddles_rsts[i].name] = middles_rsts[:, i, :]
        input_feed[model.xmiddles_lens[i].name] = middles_lens[:, i]

        targets_sents = middles_sents[:, i, :]
        targets_sents = targets_sents[:, 1:]
        targets_sents = np.reshape(
            np.array([list(t) + [0] for t in targets_sents]), (batch_size, -1))
        #NOTE [0] is vocab.PAD

        targets_rsts = middles_rsts[:, i, :]
        targets_rsts = targets_rsts[:, 1:]
        targets_rsts = np.reshape(
            np.array([list(t) + [0] for t in targets_rsts]), (batch_size, -1))

        targets_lens = middles_lens[:, i]
        targets_lens = targets_lens[1:].tolist() + [0]
        targets_lens = np.reshape(targets_lens, (batch_size))

        input_feed[model.targets_sents[i].name] = targets_sents
        input_feed[model.targets_rsts[i].name] = targets_rsts
        input_feed[model.targets_lens[i].name] = targets_lens

    return input_feed


def run(session, model, reader, verbose=False):
    #state = model.get_initial_state()
    total_loss, total_loss_lm, total_loss_crf, total_word_cnt = 0, 0, 0, 0
    start_time = time.time()

    for batch_id, batch_num, word_cnt, batch in reader.yieldSpliceBatch(
            model.mode, model.batch_size, model.step_size, para_len=model.para_len, shuffle=True):

        #print (' '.join(['%s:[%s]'%(k,'x'.join([str(s) for s in v.shape])) for k,v in batch.items() if type(v)!=int ]))
        state = model.get_initial_state()
        input_feed = load_input_feed(batch, model, state, model.batch_size)

        loss, loss_lm, loss_crf, _, summary = session.run(
            [model.loss, model.loss_lm, model.loss_crf, model.train_op, model.summary_op], input_feed)

        model.summary_writer.add_summary(
            summary, model.global_step.eval() * batch_num + batch_id)

        total_loss += loss
        total_loss_lm += loss_lm
        total_loss_crf += loss_crf
        total_word_cnt += word_cnt

        ppl = np.exp(total_loss_lm / total_word_cnt)
        wps = total_word_cnt / (time.time() - start_time)

        progress(
            batch_id *
            1.0 /
            batch_num,
            'loss: %.2f (lm:%.2f crf:%.2f) ppl: %.2f speed: %.0f wps words %d' %
            (total_loss /
             total_word_cnt,
             total_loss_lm /
             total_word_cnt,
             total_loss_crf /
             total_word_cnt,
             ppl,
             wps,
             total_word_cnt))
    print()
    return total_loss, total_word_cnt, np.exp(total_loss_lm / total_word_cnt)


def train():
    config, MODEL = LOAD_CONFIG_MODEL(FLAGS)
    pprint.pprint(FLAGS.__flags)
    pprint.pprint(dict(vars(config)))

    reader = LOAD_READER(FLAGS, config)
    tf_config = LOAD_TF_CONFIG()

    # re-setting vocab_size
    setattr(config, 'vocab_size', reader.vocab.size())
    print('Vocab size:', config.vocab_size)

    # embedding
    embedding_var = load_embedding(
        config,
        FLAGS.embedding_path,
        reader.vocab.word2id)
    lr_updater = LearningRateUpdater(
        config.learning_rate,
        config.decay,
        config.decay_when)
    graph = tf.Graph()
    with graph.as_default():
        train_model = MODEL(
            config,
            mode="train",
            pretrained_embeddings=embedding_var,
            reuse=False)
        train_model.init()
        valid_model = MODEL(
            config,
            mode="valid",
            pretrained_embeddings=embedding_var,
            reuse=True)
        valid_model.init()
        test_model = MODEL(
            config,
            mode="test",
            pretrained_embeddings=embedding_var,
            reuse=True)
        test_model.init()

    with tf.Session(graph=graph, config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        train_model.initialize(sess)

        if config.use_embeddings:
            sess.run([train_model.embedding_init], feed_dict={
                     train_model.embedding_placeholder: embedding_var})
            del embedding_var

        train_ppls, valid_ppls, test_ppls = [], [], []
        for epoch in range(train_model.global_step.eval(), config.epoch_num):
            train_model.update_lr(sess, lr_updater.get_lr())
            train_model.global_step.assign(epoch).eval()
            INFO_LOG(
                "Epoch {}, learning rate: {}".format(
                    epoch + 1, lr_updater.get_lr()))

            # train
            cost, word_cnt, ppl = run(sess, train_model, reader)
            INFO_LOG(
                "Epoch %d Train perplexity %.3f words %d" %
                (epoch + 1, ppl, word_cnt))
            train_ppls.append(ppl)

            # valid
            cost, word_cnt, ppl = run(sess, valid_model, reader)
            INFO_LOG(
                "Epoch %d Valid perplexity %.3f words %d" %
                (epoch + 1, ppl, word_cnt))
            # model hook and save the best model
            if len(valid_ppls) > 0 and ppl < min(valid_ppls):
                train_model.save(sess, global_step=epoch)
            valid_ppls.append(ppl)
            lr_updater.update(ppl)

            # test
            cost, word_cnt, ppl = run(sess, test_model, reader)
            INFO_LOG(
                "Epoch %d Test perplexity %.3f words %d" %
                (epoch + 1, ppl, word_cnt))
            test_ppls.append(ppl)
            INFO_LOG("")

            # early stopping
            if len(valid_ppls) > 5 and all(
                    valid_ppls[-1] > valid_ppls[-2:-6:-1]):
                print('Early Stopping...', valid_ppls[-1:-6:-1])
                break

        fout = open('log.txt', 'a')
        fout.write('%s\t%s' % (FLAGS.dataset, FLAGS.model_name))
        best_valid_epoch = np.argmin(valid_ppls)
        fout.write('\t%d\t%.3f\t%.3f\t%.3f\n' % (
            best_valid_epoch,
            train_ppls[best_valid_epoch],
            valid_ppls[best_valid_epoch],
            test_ppls[best_valid_epoch],
        ))

        print('%d\t%.3f\t%.3f\t%.3f\n' % (
            best_valid_epoch,
            train_ppls[best_valid_epoch],
            valid_ppls[best_valid_epoch],
            test_ppls[best_valid_epoch],
        ))


def test_ranking(session, model, reader, verbose=False, fwrite=False):
    ranks = []
    for batch_id, batch_num, word_cnt, batch in reader.yieldSpliceBatch(
            model.mode, model.batch_size, model.step_size, shuffle=False):

        rank_lables = np.empty([model.batch_size, model.para_len - 2])
        selected_idxs = [[] for _ in range(model.batch_size)]
        selected_states = [[] for _ in range(model.batch_size)]

        for level in range(model.para_len - 2 - 1):
            likelihoods_level = []
            states_level = []

            if level == 0:
                state = model.get_initial_state()
            else:
                state = np.array([s[-1] for s in selected_states]
                                 ).reshape(1, 2, model.batch_size, -1)

            states_middle = [[] for _ in range(model.batch_size)]
            for middle in range(model.para_len - 2):
                probs_middle = [[] for _ in range(model.batch_size)]
                for pid in range(model.step_size):
                    if pid == 0:
                        input_feed = {
                            model.xfirst: batch['xfirst'],
                            model.xlast: batch['xlast'],
                            model.initial_state: state}
                        for i in range(model.para_len - 2):
                            middle_start = np.empty(
                                [model.batch_size, model.step_infer_size])
                            middle_start.fill(
                                reader.vocab.word2id[reader.vocab.BOS])
                            input_feed[model.xmiddles[i].name] = middle_start
                    else:
                        input_feed = {
                            model.enc_states[0]: states[level]}
                        for i in range(model.para_len - 2):
                            input_feed[model.xmiddles[i].name] = np.reshape(
                                batch['xmiddles'][:, middle, pid], [model.batch_size, model.step_infer_size])

                    preds, logits, states, _ = session.run(
                        [model.preds, model.logits, model.final_states, model.train_op], input_feed)
                    # calculate likelihood except first BOS letter
                    if pid == 0:
                        continue

                    states_first = states[level]
                    logits_first = np.reshape(
                        logits[level], [
                            model.batch_size, model.step_infer_size, -1])
                    # step_infer_size for testing is always 1
                    idx_first = np.argmax(logits_first, axis=-1)
                    prob_first = np.take(logits_first, idx_first)
                    for i in range(model.batch_size):
                        probs_middle[i].append(prob_first[i])

                for i in range(model.batch_size):
                    states_first_one = tuple(
                        [tf.nn.rnn_cell.LSTMStateTuple(states_first[idx][0][i], states_first[idx][1][i])
                         for idx in range(model.lstm_layers)])
                    states_middle[i].append(states_first_one)  # [32 x 3]
                likelihood = [np.sum(p) for p in probs_middle]
                likelihoods_level.append(likelihood)

            # choose most likelihood sentence [3 x 32]
            for batch_idx, batch_likelihoods in enumerate(
                    np.array(likelihoods_level).T):
                batch_most = (-batch_likelihoods).argsort()
                for m in batch_most:
                    if m not in selected_idxs[batch_idx]:
                        selected_idxs[batch_idx].append(m)
                        state = states_middle[batch_idx][m]
                        selected_states[batch_idx].append(state)
                        break

        label = list(set(range(model.para_len - 2)))
        for bid, idxs in enumerate(selected_idxs):
            if len(idxs) != len(set(idxs)):
                print('Wrong! Duplicate')
                sys.exit(1)
            last = set(label) - set(idxs)
            idxs += list(last)
            ndcg = ndcg_at_k(np.power(2, idxs), len(idxs), method=1)
            # [0,1,2], [2,1,0]
            #mrr = mean_reciprocal_rank()
            ranks.append(ndcg)

        progress(
            batch_id * 1.0 / batch_num, 'rank %.3f' %
            (np.average(ranks) * 100.0))
        if verbose:
            for bid in range(3):
                print('[0]', ' '.join([reader.vocab.words[wid]
                                       for wid in batch['xfirst'][bid] if wid > 2]))
                print('[1]', ' '.join([reader.vocab.words[wid]
                                       for wid in batch['xmiddles'][bid][0] if wid > 2]))
                print('[2]', ' '.join([reader.vocab.words[wid]
                                       for wid in batch['xmiddles'][bid][1] if wid > 2]))
                print('[3]', ' '.join([reader.vocab.words[wid]
                                       for wid in batch['xlast'][bid] if wid > 2]))
                for level in range(2):
                    print(
                        '[PRED%d]' %
                        (level), ' '.join(
                            middles_preds[level][bid]))
                print()

    # evaluate bridging
    final_rank = np.average(ranks)
    print('\nNDCG\t%.4f' % (final_rank * 100.0))
    if fwrite:
        fwrite.write(
            '%s\t%s\t%s\t\n' %
            (FLAGS.dataset,
             FLAGS.model_name,
             'ranking'))
        fwrite.write('NDCG\t%.4f\n' % (final_rank * 100.0))
    print()




def test_bridging(session, model, reader, verbose=False, fwrite=False):
    middles, gens = [], []
    for batch_id, batch_num, word_cnt, batch in reader.yieldSpliceBatch(
            model.mode, model.batch_size, model.step_size, shuffle=False):
        state = model.get_initial_state()
        middles_preds = []
        for level in range(model.para_len - 2):
            middle_preds = [[] for _ in range(model.batch_size)]
            for pid in range(model.step_size):
                if pid == 0:
                    if level == 0:
                        input_feed = {
                            model.xfirst: batch['xfirst'],
                            model.xlast: batch['xlast'],
                            model.initial_state: state}
                    else:
                        input_feed = {
                            model.enc_states[0]: states[1]}
                    for i in range(model.para_len - 2):
                        middle_start = np.empty(
                            [model.batch_size, model.step_infer_size])
                        middle_start.fill(
                            reader.vocab.word2id[reader.vocab.BOS])
                        input_feed[model.xmiddles[i].name] = middle_start
                else:
                    input_feed = {
                        model.enc_states[0]: states[1]}
                    for i in range(model.para_len - 2):
                        input_feed[model.xmiddles[i].name] = preds[i]

                preds, states, _ = session.run(
                    [model.preds, model.final_states, model.eval_op], input_feed)
                # step_infer_size for testing is always 1
                preds = np.reshape(
                    preds, [
                        model.para_len - 2, model.batch_size, model.step_infer_size, -1])
                preds = np.argmax(preds, axis=-1)
                for bid in range(model.batch_size):
                    middle_preds[bid].append(
                        ' '.join([reader.vocab.words[wid] for wid in preds[0][bid] if wid > 2]))
            middles_preds.append(middle_preds)

        progress(batch_id * 1.0 / batch_num, '')

        if verbose:
            for bid in range(3):
                print('[0]', ' '.join([reader.vocab.words[wid]
                                       for wid in batch['xfirst'][bid] if wid > 2]))
                print('[1]', ' '.join([reader.vocab.words[wid]
                                       for wid in batch['xmiddles'][bid][0] if wid > 2]))
                print('[2]', ' '.join([reader.vocab.words[wid]
                                       for wid in batch['xmiddles'][bid][1] if wid > 2]))
                print('[3]', ' '.join([reader.vocab.words[wid]
                                       for wid in batch['xlast'][bid] if wid > 2]))
                for level in range(2):
                    print(
                        '[PRED%d]' %
                        (level), ' '.join(
                            middles_preds[level][bid]))
                print()

        # evaluate bridging
        for bid in range(model.batch_size):
            for i in range(model.para_len - 2):
                middles.append(' '.join([reader.vocab.words[wid]
                                         for wid in batch['xmiddles'][bid][i] if wid > 2]))
                gens.append(' '.join(middles_preds[i][bid]))

    # evaluate bridging
    print('\n')
    if fwrite:
        fwrite.write(
            '%s\t%s\t%s\t\n' %
            (FLAGS.dataset,
             FLAGS.model_name,
             'bridging'))
    results = compute_metrics_all(middles, gens)
    for m, v in results.items():
        print('%s\t%.4f' % (m, v))
        if fwrite:
            fwrite.write('%s\t%.4f\n' % (m, v))





def test():
    config, MODEL = LOAD_CONFIG_MODEL(FLAGS)
    pprint.pprint(FLAGS.__flags)
    pprint.pprint(dict(vars(config)))

    reader = LOAD_READER(FLAGS, config)
    tf_config = LOAD_TF_CONFIG()

    fwrite = open('log_test.txt', 'a')

    graph = tf.Graph()
    with graph.as_default():
        train_model = MODEL(config, mode="train", reuse=False)
        valid_model = MODEL(config, mode="valid", reuse=True)
        test_model = MODEL(config, mode="test", reuse=True)
        train_model.init()
        valid_model.init()
        test_model.init()

    with tf.Session(graph=graph, config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        train_model.initialize(sess)

        cost, word_cnt, ppl = run(sess, test_model, reader)
        INFO_LOG("Test perplexity %.3f words %d %.2f" % (ppl, word_cnt, cost))
        briding = test_bridging
        ranking = test_ranking

        bridging(sess, test_model, reader, verbose=True, fwrite=fwrite)
        ranking(sess, test_model, reader, verbose=True, fwrite=fwrite)


if __name__ == '__main__':
    if not FLAGS.is_test:
        train()
    else:
        test()
