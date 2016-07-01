import cv2
import tensorflow as tf
import sys
from model import Model
from dataset import *
from network import *
from datetime import datetime

from tensorflow.python.framework import ops

def main():

#    train_data, train_target, driver_id, unique_drivers = \
#        read_and_normalize_and_shuffle_train_data(img_rows, img_cols,
#                                                  color_type_global)
#    #Split data
#    X_train, X_test, y_train, y_test = split_validation_set(train_data, train_target, 0.2)                                              

    # Learning params
    learning_rate = 0.001
    training_iters = 1400 # 5 epochs
#    training_iters = 71 # 10 epochs
    batch_size = 64
    display_step = 40
#    test_step = 640 # 0.5 epoch
    test_step = 280

    # Network params
    n_classes = 10
    keep_rate = 0.5
    fold = 1
    
    #K-fold
    kf = KFold(26, 10, shuffle=True, random_state=20)
    
    for train_index, test_index in kf:
        print("TRAIN:", train_index, "TEST:", test_index)
        
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
#            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
            optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.9, 
                                                  momentum=0.9, epsilon= 1.0).minimize(loss)
        
            # Evaluation
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
            # Init
            init = tf.initialize_all_variables()
            
            #Saver
            saver = tf.train.Saver()

        # Load dataset
        dataset = Dataset(train_index, test_index)
    
        # Launch the graph
        with tf.Session(graph=graph) as sess:
            print 'Init variable'
            sess.run(init)
            
            # Load pretrained model
            load_with_skip('vggnet.npy', sess, ['fc8']) # Skip weights from fc8
            
            print 'Start training'
            step = 1
            while step < training_iters:
    #            offset = (step * batch_size) % (y_train.shape[0] - batch_size)
    #            batch_xs = X_train[offset:(offset + batch_size), :, :, :]
    #            batch_ys = y_train[offset:(offset + batch_size), :]
                batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')
    #            print(batch_xs.shape)
    #            print(batch_ys.shape)
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})
               
                # Display testing status
                if step%test_step == 0:
                    test_acc = 0.
                    test_count = 0
                    for _ in range(int(dataset.test_size/batch_size)):
    #                    offset = (step * batch_size) % (y_test.shape[0] - batch_size)
    #                    batch_tx = X_train[offset:(offset + batch_size), :, :, :]
    #                    batch_ty = y_train[offset:(offset + batch_size), :]
                        batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
                        acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                        test_acc += acc
                        test_count += 1
                    test_acc /= test_count
                    print >> sys.stderr, "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc)
    
    
                # Display training status
                if step%display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                    batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                    print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc)
         
                step += 1
#            tf.reset_default_graph()
            print "Finish!"
            save_path = saver.save(sess, "model" +str(fold) +".ckpt")
            print("Model saved in file: %s" % save_path)
            fold+=1
#        break

if __name__ == '__main__':
    main()

