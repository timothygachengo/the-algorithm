# checkstyle: noqa
import tensorflow.compat.v1 as tf
from ..constants import EB_SCORE_IDX

# The rationale behind this logic is available at TQ-9678.
def get_lolly_logits(labels):
  '''
  :param labels: tf.Tensor of shape (batch size, num labels) with labels as specified by the feature config.
  :return: tf.Tensor of shape (batch size) with the extracted lolly logits.
  '''
  eb_lolly_scores = get_lolly_scores(labels)
  inverse_eb_lolly_scores = tf.math.subtract(1.0, eb_lolly_scores)
  return tf.math.subtract(tf.math.log(eb_lolly_scores),
                          tf.math.log(inverse_eb_lolly_scores))

def get_lolly_scores(labels):
  '''
  :param labels: tf.Tensor of shape (batch size, num labels) with labels as specified by the feature config.
  :return: tf.Tensor of shape (batch size) with the extracted lolly scores.
  '''
  logged_eb_lolly_scores = tf.reshape(labels[:, EB_SCORE_IDX], (-1, 1))
  return tf.truediv(logged_eb_lolly_scores, 100.0)
