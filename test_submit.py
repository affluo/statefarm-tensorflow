import cv2
import tensorflow as tf
import sys
from model import Model
from testset import *
from network import *
from datetime import datetime

from tensorflow.python.framework import ops
                                        
training_iters = 1247 
batch_size = 64
display_step = 40
#    test_step = 640 # 0.5 epoch
test_step = 280

# Network params
n_classes = 10
keep_rate = 0.5
fold = 1

fold_pred = []

kf = KFold(26, 10, shuffle=True, random_state=20)
    
for train_index, test_index in kf:

    
    graph = tf.Graph()
    with graph.as_default():
    
        # Graph input
        x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
        y = tf.placeholder(tf.float32, [None, n_classes])
        keep_var = tf.placeholder(tf.float32)
    
        # Model
        pred = Model.vggnet(x, keep_var)
    
        # Loss and optimizer
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    
        # Evaluation
        test_prediction = tf.nn.softmax(pred)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
        # Init
        init = tf.initialize_all_variables()
        
        #Saver
        saver = tf.train.Saver()
    

#test_list, test_id = testset.load_test()

#    batch_xs = testset.next_batch(1, 'test')
    
    # Load dataset
    testset = Testset()

    # Launch the graph
    predictions = []
    with tf.Session(graph=graph) as sess:
        print 'Init variable'
        sess.run(init)
        
        # Load pretrained model
    #        load_with_skip('vggnet.npy', sess, ['fc8']) # Skip weights from fc8
        model_line = "model" + str(fold) + ".ckpt"
        saver.restore(sess, model_line)
        print("Model restored.")
        
        print 'Start testing'
        step = 1
        while step < training_iters:
    #            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
    #            batch_xs = X_train[offset:(offset + batch_size), :, :, :]
    #            batch_ys = y_train[offset:(offset + batch_size), :]
            batch_xs = testset.next_batch(batch_size, 'test')
    #            print(batch_xs.shape)
    #            print(batch_ys.shape)
    #            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})
            batch_prediction = sess.run(test_prediction, feed_dict={x: batch_xs, keep_var: 1.})
            batch_prediction = batch_prediction.tolist()
            predictions = predictions + batch_prediction
    #            print(prediction.shape)
    #            break
            # Display testing status
            if step%test_step == 0:
                print('Tested: '+ str(batch_size*step))
    #                test_acc = 0.
    #                test_count = 0
    #                for _ in range(int(testset.test_size/batch_size)):
    ##                    offset = (step * batch_size) % (y_test.shape[0] - batch_size)
    ##                    batch_tx = X_train[offset:(offset + batch_size), :, :, :]
    ##                    batch_ty = y_train[offset:(offset + batch_size), :]
    #                    batch_tx = testset.next_batch(batch_size, 'test')
    #                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
    #                    test_acc += acc
    #                    test_count += 1
    #                test_acc /= test_count
    #                print >> sys.stderr, "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc)
    
    
    #            # Display training status
    #            if step%display_step == 0:
    #                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
    #                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
    #                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc)
     
            step += 1
    #            tf.reset_default_graph()
    #        break
        print "Finish!"
        predictions = predictions[0:79726]
        fold_pred.append(predictions)
        fold+=1
    

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()
    
info = "8_folds_base"
test_list, test_id = testset.load_test()
    
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3',
                                                 'c4', 'c5', 'c6', 'c7',
                                                 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1 = result1.sort_values(by='img')
    cols = result1.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    result1 = result1[cols]    
    now = datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)
    
    return result1
    
final_pred = merge_several_folds_mean(fold_pred, 10)
result = create_submission(final_pred, test_id, info)

#if __name__ == '__main__':
#    main()

