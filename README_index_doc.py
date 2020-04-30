mlmodels\benchmark.py
----------------methods----------------

---------------functions---------------
benchmark_run( bench_pars=None, args=None, config_mode="test",  )
cli_load_arguments( config_file=None,  )
config_model_list( folder=None,  )
get_all_json_path(json_path,   )
main(  )
metric_eval( actual=None, pred=None, metric_name="mean_absolute_error",  )


mlmodels\data.py
----------------methods----------------

---------------functions---------------
download_data(data_pars,   )
get_dataset(data_pars,   )
import_data(  )
import_data_dask(  **kw)
import_data_fromfile(  **kw)
import_data_tch( name="", mode="train", node_id=0, data_folder_root="",  )
tf_dataset(dataset_pars,   )


mlmodels\dataloader.py
----------------methods----------------
DataLoader.__init__(self, data_pars,   )
DataLoader.compute(self,   )
DataLoader.get_data(self,  intermediate=False,  )

---------------functions---------------
_check_output_shape(self, inter_output, shape, max_len,   )
_interpret_input_pars(self, input_pars,   )
batch_generator(iterable,  n=1,  )
image_dir_load(path,   )
pickle_dump(t, path,   )
pickle_load(file,   )
split_xy_from_dict(out, data_pars,   )


mlmodels\distri_torch.py
----------------methods----------------

---------------functions---------------
load_arguments(  )
metric_average(val, name,   )
test(  )
train(epoch,   )


mlmodels\metrics.py
----------------methods----------------

---------------functions---------------
log( n=0, m=1,  *s)
metrics_eval( metric_list=["mean_squared_error"], ytrue=None, ypred=None, ypred_proba=None, return_dict=0,  )
test(  )


mlmodels\models.py
----------------methods----------------

---------------functions---------------
cli_load_arguments( config_file=None,  )
config_generate_json(modelname,  to_path="ztest/new_model/",  )
config_get_pars(config_file,  config_mode="test",  )
config_init( to_path=".",  )
config_model_list( folder=None,  )
fit(module, model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
fit_metrics(module, model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
get_params(module, params_pars,   **kwarg)
load(module, load_pars,   **kwarg)
main(  )
metrics(module, model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
model_create(module,  model_pars=None, data_pars=None, compute_pars=None,  **kwarg)
module_env_build( model_uri="", verbose=0, do_env_build=0,  )
module_load( model_uri="", verbose=0, env_build=0,  )
module_load_full( model_uri="", model_pars=None, data_pars=None, compute_pars=None, choice=None,  **kwarg)
os_folder_copy(src, dst,   )
predict(module, model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
save(module, model, session, save_pars,   **kwarg)
test(modelname,   )
test_all( folder=None,  )
test_api( model_uri="model_xxxx/yyyy.py", param_pars=None,  )
test_global(modelname,   )
test_module( model_uri="model_xxxx/yyyy.py", param_pars=None,  )


mlmodels\optim.py
----------------methods----------------

---------------functions---------------
cli_load_arguments( config_file=None,  )
main(  )
optim( model_uri="model_tf.1_lstm.py", hypermodel_pars={}, model_pars={}, data_pars={}, compute_pars={}, out_pars={},  )
optim_optuna( model_uri="model_tf.1_lstm.py", hypermodel_pars={"engine_pars": {},  )
post_process_best(model, module, model_uri, model_pars_update, data_pars, compute_pars, out_pars,   )
test_all(  )
test_fast( ntrials=2,  )
test_json( path_json="", config_mode="test",  )


mlmodels\parse.py
----------------methods----------------

---------------functions---------------
cli_load_arguments( config_file=None,  )
extract_args(txt, outfile,   )


mlmodels\pipeline.py
----------------methods----------------
Pipe.__init__(self, pipe_list, in_pars, out_pars,  compute_pars=None,  **kw)
Pipe.get_checkpoint(self,   )
Pipe.get_fitted_pipe_list(self,  key="",  )
Pipe.get_model_path(self,   )
Pipe.run(self,   )

---------------functions---------------
drop_cols(df,  cols=None,  **kw)
generate_data(df,  num_data=0, means=[], cov=[[1, 0],  )
get_params( choice="", data_path="dataset/", config_mode="test",  **kw)
load_model(path,   )
log( n=0, m=1,  *s)
os_package_root_path(filepath,  sublevel=0, path_add="",  )
pd_concat(df1, df2, colid1,   )
pd_na_values(df,  cols=None, default=0.0,  **kw)
pipe_checkpoint(df,   **kw)
pipe_load(df,   **in_pars)
pipe_merge(in_pars, out_pars,  compute_pars=None,  **kw)
pipe_run_inference(pipe_list, in_pars, out_pars,  compute_pars=None, checkpoint=True,  **kw)
pipe_split(in_pars, out_pars, compute_pars,   **kw)
save_model(model, path,   )
test( data_path="/dataset/", pars_choice="colnum",  )


mlmodels\preprocessor.py
----------------methods----------------
MissingDataPreprocessorError.__init__(self,   )
Preprocessor.__init__(self, preprocessor_dict,   )
Preprocessor._interpret_preprocessor_dict(self, pars,   )
Preprocessor._name_outputs(self, names, outputs,   )
Preprocessor.fit_transform(self, data,   )
Preprocessor.transform(self, data,   )
PreprocessorNotFittedError.__init__(self,   )

---------------functions---------------


mlmodels\test_dataloader.py
----------------methods----------------
SingleFunctionPreprocessor.__init__(self, func_dict,   )
SingleFunctionPreprocessor.compute(self, data,   )
SingleFunctionPreprocessor.get_data(self,   )

---------------functions---------------
gluon_append_target_string(out, data_pars,   )
identical_test_set_split(test_size,   *args, **kwargs)
load_npz(path,   )
pandas_load_train_test(path, test_path,   **args)
pandas_split_xy(out, data_pars,   )
read_csvs_from_directory(path,  files=None,  **args)
rename_target_to_y(out, data_pars,   )
split_timeseries_df(out, data_pars, length, shift,   )
split_xy_from_dict(out, data_pars,   )
timeseries_split( test_size=0.2,  *args)
tokenize_x(data, no_classes,  max_words=None,  )


mlmodels\util.py
----------------methods----------------
Model_empty.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
to_namespace.__init__(self, adict,   )
to_namespace.get(self, key,   )

---------------functions---------------
config_load_root(  )
config_path_dataset(  )
config_path_pretrained(  )
config_set(ddict2,   )
env_build(model_uri, env_pars,   )
env_conda_build( env_pars=None,  )
env_pip_check( env_pars=None,  )
env_pip_requirement( env_pars=None,  )
get_device_torch(  )
get_model_uri(file,   )
get_recursive_files(folderPath,  ext='/*model*/*.py',  )
get_recursive_files2(folderPath, ext,   )
get_recursive_files3(folderPath, ext,   )
json_norm(ddict,   )
load(load_pars,   )
load_callable_from_dict(function_dict,  return_other_keys=False,  )
load_callable_from_uri(uri,   )
load_config(args, config_file, config_mode,  verbose=0,  )
load_function( package="mlmodels.util", name="path_norm",  )
load_gluonts( load_pars=None,  )
load_keras(load_pars,  custom_pars=None,  )
load_pkl(load_pars,   )
load_tch(load_pars,   )
load_tch_checkpoint(model, optimiser, load_pars,   )
load_tf( load_pars="",  )
log( n=0, m=1,  *s)
model_get_list( folder=None, block_list=[],  )
os_file_current_path(  )
os_folder_copy(src, dst,   )
os_get_file( folder=None, block_list=[], pattern=r'*.py',  )
os_package_root_path( filepath="", sublevel=0, path_add="",  )
os_path_split(path,   )
params_json_load(path,  config_mode="test", tlist=[ "model_pars", "data_pars", "compute_pars", "out_pars"],  )
path_norm( path="",  )
path_norm_dict(ddict,   )
save( model=None, session=None, save_pars=None,  )
save_gluonts( model=None, session=None, save_pars=None,  )
save_keras( model=None, session=None, save_pars=None,  )
save_pkl( model=None, session=None, save_pars=None,  )
save_tch( model=None, optimizer=None, save_pars=None,  )
save_tch_checkpoint(model, optimiser, save_pars,   )
save_tf( model=None, sess=None, save_pars=None,  )
test_module( model_uri="model_tf/1_lstm.py", data_path="dataset/", pars_choice="json", reset=True,  )
tf_deprecation(  )
val(x, xdefault,   )


mlmodels\util_log.py
----------------methods----------------
to_name.__init__(self, adict,   )

---------------functions---------------
create_appid(filename,   )
create_logfilename(filename,   )
create_uniqueid(  )
load_arguments( config_file=None, arg_list=None,  )
logger_handler_console( formatter=None,  )
logger_handler_file( isrotate=False, rotate_time="midnight", formatter=None, log_file_used=None,  )
logger_setup( logger_name=None, log_file=None, formatter=FORMATTER_1, isrotate=False, isconsole_output=True, logging_level=logging.DEBUG,  )
logger_setup2( name=__name__, level=None,  )
printlog( s="", s1="", s2="", s3="", s4="", s5="", s6="", s7="", s8="", s9="", s10="", app_id="", logfile=None, iswritelog=True,  )
writelog( m="", f=None,  )


mlmodels\ztest.py
----------------methods----------------

---------------functions---------------
cli_load_arguments( config_file=None,  )
main(  )
os_file_current_path(  )
test_all( arg=None,  )
test_benchmark( arg=None,  )
test_custom(  )
test_import(arg,   )
test_json(arg,   )
test_jupyter( arg=None, config_mode="test_all",  )
test_list(mlist,   )
test_model_structure(  )


mlmodels\ztest_structure.py
----------------methods----------------

---------------functions---------------
code_check( sign_list=None, model_list=None,  )
find_in_list(x, llist,   )
get_recursive_files(folderPath,  ext='/*model*/*.py',  )
log( n=0, m=1,  *s)
main(  )
model_get_list( folder=None, block_list=[],  )
os_file_current_path(  )
os_package_root_path(filepath,  sublevel=0, path_add="",  )


mlmodels\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\config\json\model_tch\raw\vae_pretraining_encoder\text_beta.py
----------------methods----------------

---------------functions---------------


mlmodels\example\arun_hyper.py
----------------methods----------------

---------------functions---------------


mlmodels\example\arun_model.py
----------------methods----------------

---------------functions---------------


mlmodels\example\benchmark_timeseries_m4.py
----------------methods----------------

---------------functions---------------
benchmark_m4(  )


mlmodels\example\benchmark_timeseries_m5.py
----------------methods----------------
M5Evaluator.get_aggregate_metrics(self, metric_per_ts,   )
M5Evaluator.get_metrics_per_ts(self, time_series, forecast,   )

---------------functions---------------
plot_prob_forecasts(ts_entry, forecast_entry, path, sample_id,  inline=True,  )


mlmodels\example\lightgbm_glass.py
----------------methods----------------

---------------functions---------------


mlmodels\example\vision_mnist.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\dev\ml_mosaic.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\dev\mytest.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\raw\abstractive-summarization\pointer_generator_helper.py
----------------methods----------------
PointerGeneratorAttentionWrapper.__init__(self, cell, attention_mechanism,  attention_layer_size=None, alignment_history=False, cell_input_fn=None, output_attention=True, initial_cell_state=None, name=None, coverage=False,  )
PointerGeneratorAttentionWrapper.call(self, inputs, state,   )
PointerGeneratorAttentionWrapper.zero_state(self, batch_size, dtype,   )
PointerGeneratorBahdanauAttention.__call__(self, query, state,   )
PointerGeneratorBahdanauAttention.__init__(self, num_units, memory,  memory_sequence_length=None, normalize=False, probability_fn=None, score_mask_value=float('-inf'), name='PointerGeneratorBahdanauAttention', coverage=False,  )
PointerGeneratorDecoder.__init__(self, source_extend_tokens, source_oov_words, coverage, cell, helper, initial_state,  output_layer=None,  )
PointerGeneratorDecoder.output_dtype(self,   )
PointerGeneratorDecoder.output_size(self,   )
PointerGeneratorDecoder.step(self, time, inputs, state,  name=None,  )
PointerGeneratorGreedyEmbeddingHelper.__init__(self, embedding, start_tokens, end_token,   )
PointerGeneratorGreedyEmbeddingHelper.next_inputs(self, time, outputs, state, sample_ids,  name=None,  )
PointerGeneratorGreedyEmbeddingHelper.sample(self, time, outputs, state,  name=None,  )

---------------functions---------------
_pg_bahdanau_score(processed_query, keys, coverage, coverage_vector,   )


mlmodels\model_dev\raw\chatbot\35.byte-net.py
----------------methods----------------
ByteNet.__init__(self, from_vocab_size, to_vocab_size, channels, encoder_dilations, decoder_dilations, encoder_filter_width, decoder_filter_width,  learning_rate=0.001, beta1=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width,  causal=True,  )
clean_text(text,   )
conv1d(input_, output_channels,  dilation=1, filter_width=1, causal=False,  )
layer_normalization(x,  epsilon=1e-8,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_dev\raw\chatbot\37.capsule-lstm-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, seq_len, maxlen, from_dict_size, to_dict_size, learning_rate, batch_size,  kernels=[2, 4, 4], strides=[3,2,1], epsilon=1e-8,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pad_sentence_batch_static(sentence_batch, pad_int,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )
str_idx(corpus, dic,   )


mlmodels\model_dev\raw\chatbot\38.capsule-lstm-seq2seq-luong-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, seq_len, maxlen, from_dict_size, to_dict_size, learning_rate, batch_size,  kernels=[2, 4, 4], strides=[3,2,1], epsilon=1e-8, force_teaching_ratio=0.5, beam_width=5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pad_sentence_batch_static(sentence_batch, pad_int,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )
str_idx(corpus, dic,   )


mlmodels\model_dev\raw\chatbot\access.py
----------------methods----------------
MemoryAccess._linear(first_dim, second_dim, name,  activation=None,  )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_dev\raw\chatbot\addressing.py
----------------methods----------------

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_dev\raw\chatbot\dnc.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\raw\chatbot\gpt_2.py
----------------methods----------------

---------------functions---------------
attention_mask(nd, ns,   )
attn(x, scope, n_state,   )
block(x, scope,   )
conv1d(x, scope, nf,   )
expand_tile(value, size,   )
gelu(x,   )
merge_states(x,   )
mlp(x, scope, n_state,   )
model(hparams, X,  past=None, scope='model', reuse=False,  )
norm(x, scope,   )
past_shape(  )
positions_for(tokens, past_length,   )
shape_list(x,   )
softmax(x,  axis=-1,  )
split_states(x, n,   )


mlmodels\model_dev\raw\chatbot\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )


mlmodels\model_dev\raw\extractive-summarization\modeling.py
----------------methods----------------

---------------functions---------------
assert_rank(tensor, expected_rank,  name=None,  )
attention_layer(from_tensor, to_tensor,  attention_mask=None, num_attention_heads=1, size_per_head=512, query_act=None, key_act=None, value_act=None, attention_probs_dropout_prob=0.0, initializer_range=0.02, do_return_2d_tensor=False, batch_size=None, from_seq_length=None, to_seq_length=None,  )
create_attention_mask_from_input_mask(from_tensor, to_mask,   )
create_initializer( initializer_range=0.02,  )
dropout(input_tensor, dropout_prob,   )
embedding_lookup(input_ids, vocab_size,  embedding_size=128, initializer_range=0.02, word_embedding_name="word_embeddings", use_one_hot_embeddings=False,  )
embedding_postprocessor(input_tensor,  use_token_type=False, token_type_ids=None, token_type_vocab_size=16, token_type_embedding_name="token_type_embeddings", use_position_embeddings=True, position_embedding_name="position_embeddings", initializer_range=0.02, max_position_embeddings=512, dropout_prob=0.1,  )
gelu(x,   )
get_activation(activation_string,   )
get_assignment_map_from_checkpoint(tvars, init_checkpoint,   )
get_shape_list(tensor,  expected_rank=None, name=None,  )
layer_norm(input_tensor,  name=None,  )
layer_norm_and_dropout(input_tensor, dropout_prob,  name=None,  )
reshape_from_matrix(output_tensor, orig_shape_list,   )
reshape_to_matrix(input_tensor,   )
transformer_model(input_tensor,  attention_mask=None, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, intermediate_act_fn=gelu, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, initializer_range=0.02, do_return_all_layers=False,  )


mlmodels\model_dev\raw\neural-machine-translation\access.py
----------------methods----------------
MemoryAccess._linear(first_dim, second_dim, name,  activation=None,  )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_dev\raw\neural-machine-translation\addressing.py
----------------methods----------------

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_dev\raw\neural-machine-translation\dnc.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\raw\neural-machine-translation\gpt_2.py
----------------methods----------------

---------------functions---------------
attention_mask(nd, ns,   )
attn(x, scope, n_state,   )
block(x, scope,   )
conv1d(x, scope, nf,   )
expand_tile(value, size,   )
gelu(x,   )
merge_states(x,   )
mlp(x, scope, n_state,   )
model(hparams, X,  past=None, scope='model', reuse=False,  )
norm(x, scope,   )
past_shape(  )
positions_for(tokens, past_length,   )
shape_list(x,   )
softmax(x,  axis=-1,  )
split_states(x, n,   )


mlmodels\model_dev\raw\neural-machine-translation\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )


mlmodels\model_dev\raw\question-answer\attention_gru.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\raw\stemming\access.py
----------------methods----------------
MemoryAccess._linear(first_dim, second_dim, name,  activation=None,  )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_dev\raw\stemming\addressing.py
----------------methods----------------

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_dev\raw\stemming\dnc.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\raw\stemming\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )


mlmodels\model_dev\raw\text-augmentation\6.vae-varitional-bahdanau\attention_wrapper.py
----------------methods----------------
AttentionMechanism._maybe_mask(m, seq_len_mask,   )
AttentionWrapper.__init__(self, cell, attention_mechanism,  temperature=1.0, use_hmean=True, attention_layer_size=None, alignment_history=False, cell_input_fn=None, output_attention=True, initial_cell_state=None, name=None,  )
AttentionWrapper._batch_size_checks(self, batch_size, error_message,   )
AttentionWrapper._item_or_tuple(self, seq,   )
AttentionWrapper.call(self, inputs, state,   )
AttentionWrapper.output_size(self,   )
AttentionWrapper.state_size(self,   )
AttentionWrapper.zero_state(self, batch_size, dtype,   )
AttentionWrapperState.clone(self,   **kwargs)
BahdanauAttention.__call__(self, query, previous_alignments,   )
BahdanauAttention.__init__(self, num_units, memory,  memory_sequence_length=None, normalize=False, probability_fn=None, score_mask_value=None, dtype=None, name="BahdanauAttention",  )
BahdanauMonotonicAttention.__call__(self, query, previous_alignments,   )
BahdanauMonotonicAttention.__init__(self, num_units, memory,  memory_sequence_length=None, normalize=False, score_mask_value=None, sigmoid_noise=0., sigmoid_noise_seed=None, score_bias_init=0., mode="parallel", dtype=None, name="BahdanauMonotonicAttention",  )
LuongAttention.__call__(self, query, previous_alignments,   )
LuongAttention.__init__(self, num_units, memory,  memory_sequence_length=None, scale=False, probability_fn=None, score_mask_value=None, dtype=None, name="LuongAttention",  )
LuongMonotonicAttention.__call__(self, query, previous_alignments,   )
LuongMonotonicAttention.__init__(self, num_units, memory,  memory_sequence_length=None, scale=False, score_mask_value=None, sigmoid_noise=0., sigmoid_noise_seed=None, score_bias_init=0., mode="parallel", dtype=None, name="LuongMonotonicAttention",  )
_BaseAttentionMechanism.__init__(self, query_layer, memory, probability_fn,  memory_sequence_length=None, memory_layer=None, check_inner_dims_defined=True, score_mask_value=None, name=None,  )
_BaseAttentionMechanism.alignments_size(self,   )
_BaseAttentionMechanism.batch_size(self,   )
_BaseAttentionMechanism.initial_alignments(self, batch_size, dtype,   )
_BaseAttentionMechanism.keys(self,   )
_BaseAttentionMechanism.memory_layer(self,   )
_BaseAttentionMechanism.query_layer(self,   )
_BaseAttentionMechanism.values(self,   )
_BaseMonotonicAttentionMechanism.initial_alignments(self, batch_size, dtype,   )

---------------functions---------------
_bahdanau_score(processed_query, keys, normalize,   )
_compute_attention(attention_mechanism, cell_output, previous_alignments, attention_layer, temperature, use_hmean,   )
_luong_score(query, keys, scale,   )
_maybe_mask_score(score, memory_sequence_length, score_mask_value,   )
_monotonic_probability_fn(score, previous_alignments, sigmoid_noise, mode,  seed=None,  )
_prepare_memory(memory, memory_sequence_length, check_inner_dims_defined,   )
hardmax(logits,  name=None,  )
monotonic_attention(p_choose_i, previous_attention, mode,   )
safe_cumprod(x,   *args, **kwargs)


mlmodels\model_dev\raw\text-augmentation\6.vae-varitional-bahdanau\basic_decoder.py
----------------methods----------------
BasicDecoder.__init__(self, cell, helper, initial_state, latent_vector,  output_layer=None,  )
BasicDecoder._rnn_output_size(self,   )
BasicDecoder.batch_size(self,   )
BasicDecoder.initialize(self,  name=None,  )
BasicDecoder.output_dtype(self,   )
BasicDecoder.output_size(self,   )
BasicDecoder.step(self, time, inputs, state,  name=None,  )

---------------functions---------------


mlmodels\model_dev\raw\text-augmentation\6.vae-varitional-bahdanau\decoder.py
----------------methods----------------
Decoder._create(s, d,   )
Decoder._t(s,   )
Decoder.batch_size(self,   )
Decoder.finalize(self, outputs, final_state, sequence_lengths,   )
Decoder.initialize(self,  name=None,  )
Decoder.output_dtype(self,   )
Decoder.output_size(self,   )
Decoder.step(self, time, inputs, state,  name=None,  )
Decoder.tracks_own_finished(self,   )

---------------functions---------------
_create_zero_outputs(size, dtype, batch_size,   )
dynamic_decode(decoder,  output_time_major=False, impute_finished=False, maximum_iterations=None, parallel_iterations=32, swap_memory=False, scope=None,  )


mlmodels\model_dev\raw\text-classification\63.deep-pyramid-cnn.py
----------------methods----------------
Model.__init__(self, maxlen, dimension_output, vocab_size, embedding_size, kernel_size, num_filters, learning_rate,   )

---------------functions---------------


mlmodels\model_dev\raw\text-classification\bert_model.py
----------------methods----------------
BertConfig.__init__(self, vocab_size,  hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act='gelu', hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02,  )
BertConfig.from_dict(cls, json_object,   )
BertConfig.from_json_file(cls, json_file,   )
BertConfig.to_dict(self,   )
BertConfig.to_json_string(self,   )
BertModel.__init__(self, config, is_training, input_ids,  input_mask=None, token_type_ids=None, use_one_hot_embeddings=True, scope=None,  )
BertModel.get_all_encoder_layers(self,   )
BertModel.get_embedding_output(self,   )
BertModel.get_embedding_table(self,   )
BertModel.get_pooled_output(self,   )
BertModel.get_sequence_output(self,   )
BertModel.transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width,   )

---------------functions---------------
assert_rank(tensor, expected_rank,  name=None,  )
attention_layer(from_tensor, to_tensor,  attention_mask=None, num_attention_heads=1, size_per_head=512, query_act=None, key_act=None, value_act=None, attention_probs_dropout_prob=0.0, initializer_range=0.02, do_return_2d_tensor=False, batch_size=None, from_seq_length=None, to_seq_length=None,  )
create_attention_mask_from_input_mask(from_tensor, to_mask,   )
create_initializer( initializer_range=0.02,  )
dropout(input_tensor, dropout_prob,   )
embedding_lookup(input_ids, vocab_size,  embedding_size=128, initializer_range=0.02, word_embedding_name='word_embeddings', use_one_hot_embeddings=False,  )
embedding_postprocessor(input_tensor,  use_token_type=False, token_type_ids=None, token_type_vocab_size=16, token_type_embedding_name='token_type_embeddings', use_position_embeddings=True, position_embedding_name='position_embeddings', initializer_range=0.02, max_position_embeddings=512, dropout_prob=0.1,  )
gelu(input_tensor,   )
get_activation(activation_string,   )
get_assignment_map_from_checkpoint(tvars, init_checkpoint,   )
get_shape_list(tensor,  expected_rank=None, name=None,  )
layer_norm(input_tensor,  name=None,  )
layer_norm_and_dropout(input_tensor, dropout_prob,  name=None,  )
reshape_from_matrix(output_tensor, orig_shape_list,   )
reshape_to_matrix(input_tensor,   )
transformer_model(input_tensor,  attention_mask=None, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, intermediate_act_fn=gelu, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, initializer_range=0.02, do_return_all_layers=False,  )


mlmodels\model_dev\raw\text-classification\dynamic_memory_network.py
----------------methods----------------
DynamicMemoryNetwork.__init__(self, num_classes, learning_rate, decay_steps, decay_rate, sequence_length, story_length, vocab_size, embed_size, hidden_size,  num_pass=2, use_gated_gru=True, decode_with_sequences=False, initializer=tf.random_normal_initializer(stddev = 0.1), clip_gradients=5.0, l2_lambda=0.0001,  )
DynamicMemoryNetwork.answer_module(self,   )
DynamicMemoryNetwork.attention_mechanism_parallel(self, c_full, m, q, i,   )
DynamicMemoryNetwork.episodic_memory_module(self,   )
DynamicMemoryNetwork.gated_gru(self, c_current, h_previous, g_current,   )
DynamicMemoryNetwork.gru_cell(self, Xt, h_t_minus_1, variable_scope,   )
DynamicMemoryNetwork.inference(self,   )
DynamicMemoryNetwork.input_module(self,   )
DynamicMemoryNetwork.instantiate_weights(self,   )
DynamicMemoryNetwork.loss(self,  l2_lambda=0.0001,  )
DynamicMemoryNetwork.question_module(self,   )
DynamicMemoryNetwork.train(self,   )
DynamicMemoryNetwork.x1Wx2_parallel(self, x1, x2, scope,   )

---------------functions---------------


mlmodels\model_dev\raw\text-classification\entity_network.py
----------------methods----------------
EntityNetwork.__init__(self, num_classes, learning_rate, decay_steps, decay_rate, sequence_length, story_length, vocab_size, embed_size, hidden_size,  block_size=20, initializer=tf.random_normal_initializer(stddev = 0.1), clip_gradients=5.0, use_bi_lstm=False,  )
EntityNetwork.activation(self, features,  scope=None,  )
EntityNetwork.cell(self, s_t, h_all, w_all, i,   )
EntityNetwork.embedding_with_mask(self,   )
EntityNetwork.inference(self,   )
EntityNetwork.input_encoder_bi_lstm(self,   )
EntityNetwork.input_encoder_bow(self,   )
EntityNetwork.instantiate_weights(self,   )
EntityNetwork.loss(self,  l2_lambda=0.0001,  )
EntityNetwork.output_module(self,   )
EntityNetwork.rnn_story(self,   )
EntityNetwork.train(self,   )

---------------functions---------------


mlmodels\model_dev\raw\text-classification\gpt_2.py
----------------methods----------------

---------------functions---------------
attention_mask(nd, ns,   )
attn(x, scope, n_state,   )
block(x, scope,   )
conv1d(x, scope, nf,   )
expand_tile(value, size,   )
gelu(x,   )
merge_states(x,   )
mlp(x, scope, n_state,   )
model(hparams, X,  past=None, scope='model', reuse=False,  )
norm(x, scope,   )
past_shape(  )
positions_for(tokens, past_length,   )
shape_list(x,   )
softmax(x,  axis=-1,  )
split_states(x, n,   )


mlmodels\model_dev\raw\text-classification\modules.py
----------------methods----------------

---------------functions---------------
attention_decoder(inputs, memory,  units=None, scope='attention_decoder', reuse=None,  )
conv1d(inputs,  filters=None, size=1, rate=1, padding='SAME', use_bias=False, activation_fn=None, scope='conv1d', reuse=None,  )
conv1d_banks(inputs,  K=16, is_training=True, scope='conv1d_banks', reuse=None,  )
embed(inputs, vocab_size, dimension,  scope='embedding', reuse=None,  )
gru(inputs,  units=None, bidirection=False, scope='gru', reuse=None,  )
highwaynet(inputs,  units=None, scope='highwaynet', reuse=None,  )
normalize_bn(inputs,  decay=0.99, is_training=True, activation_fn=None, scope='normalize_bn',  )
normalize_in(inputs,  activation_fn=None, scope='normalize_in',  )
normalize_layer_norm(inputs,  activation_fn=None, scope='normalize_layer_norm',  )
prenet(inputs,  is_training=True, scope='prenet', reuse=None,  )
shift_by_one(inputs,   )


mlmodels\model_dev\raw\text-classification\utils.py
----------------methods----------------

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
separate_dataset(trainset,  ratio=0.5,  )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_dev\raw\text-classification\xl.py
----------------methods----------------

---------------functions---------------
_cache_mem(curr_out, prev_mem,  mem_len=None,  )
_create_mask(qlen, mlen,  same_length=False,  )
embedding_lookup(lookup_table, x,   )
mask_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, proj_same_dim=True, scope='adaptive_embed',  **kwargs)
mask_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs, params, tie_projs,  initializer=None, proj_initializer=None, div_val=1, scope='adaptive_softmax', proj_same_dim=True, return_mean=True,  **kwargs)
mul_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, perms=None, proj_same_dim=True, scope='adaptive_embed',  )
mul_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs, params, tie_projs,  initializer=None, proj_initializer=None, div_val=1, perms=None, proj_same_dim=True, scope='adaptive_softmax',  **kwargs)
positional_embedding(pos_seq, inv_freq,  bsz=None,  )
positionwise_FF(inp, d_model, d_inner, kernel_initializer,  scope='ff',  )
rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model, n_head, d_head, kernel_initializer,  scope='rel_attn',  )
rel_shift(x,   )
transformer(dec_inp, mems, n_token, n_layer, d_model, d_embed, n_head, d_head, d_inner, initializer,  proj_initializer=None, mem_len=None, cutoffs=[], div_val=1, tie_projs=[], same_length=False, clamp_len=-1, untie_r=False, proj_same_dim=True, scope='transformer',  )


mlmodels\model_dev\raw\text-to-speech\caching.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\raw\text-to-speech\utils.py
----------------methods----------------

---------------functions---------------
get_cached(path,   )
get_spectrogram(audio_file,   )
get_wav(spectrogram,   )
griffin_lim(spectrogram,   )
invert_spectrogram(spectrogram,   )
load_file(path,   )
plot_alignment(alignment, e,   )
spectrogram2wav(mag,   )
text_normalize(text,   )


mlmodels\model_dev\raw\text-to-speech\1.tacotron\caching.py
----------------methods----------------

---------------functions---------------


mlmodels\model_dev\raw\text-to-speech\1.tacotron\tacotron.py
----------------methods----------------
Tacotron.__init__(self,  reuse=None,  )

---------------functions---------------
bn(inputs,  is_training=True, activation_fn=None, scope='bn', reuse=None,  )
conv1d_banks(inputs,  K=16, is_training=True, scope='conv1d_banks',  )
guided_attention( g=0.2,  )
highwaynet(inputs,  num_units=None, scope='highwaynet',  )
learning_rate_decay(init_lr, global_step,  warmup_steps=4000.0,  )
prenet(inputs,  num_units=None, is_training=True, scope='prenet',  )


mlmodels\model_dev\raw\text-to-speech\1.tacotron\utils.py
----------------methods----------------

---------------functions---------------
get_cached(path,   )
get_spectrogram(audio_file,   )
get_wav(spectrogram,   )
griffin_lim(spectrogram,   )
invert_spectrogram(spectrogram,   )
load_file(path,   )
plot_alignment(alignment,   )
spectrogram2wav(mag,   )
text_normalize(text,   )


mlmodels\model_dev\raw\topic-model\modeling.py
----------------methods----------------

---------------functions---------------
_cache_mem(curr_out, prev_mem, mem_len,  reuse_len=None,  )
_create_mask(qlen, mlen,  dtype=tf.float32, same_length=False,  )
abs_attn_core(q_head, k_head, v_head, attn_mask, dropatt, is_training, scale,   )
classification_loss(hidden, labels, n_class, initializer, scope,  reuse=None, return_logits=False,  )
embedding_lookup(x, n_token, d_embed, initializer,  use_tpu=True, scope='embedding', reuse=None, dtype=tf.float32,  )
gelu(x,   )
head_projection(h, d_model, n_head, d_head, kernel_initializer, name,   )
lm_loss(hidden, target, n_token, d_model, initializer,  lookup_table=None, tie_weight=False, bi_data=True, use_tpu=False,  )
multihead_attn(q, k, v, attn_mask, d_model, n_head, d_head, dropout, dropatt, is_training, kernel_initializer,  residual=True, scope='abs_attn', reuse=None,  )
positional_embedding(pos_seq, inv_freq,  bsz=None,  )
positionwise_ffn(inp, d_model, d_inner, dropout, kernel_initializer,  activation_type='relu', scope='ff', is_training=True, reuse=None,  )
post_attention(h, attn_vec, d_model, n_head, d_head, dropout, is_training, kernel_initializer,  residual=True,  )
regression_loss(hidden, labels, initializer, scope,  reuse=None, return_logits=False,  )
rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, is_training, scale,   )
rel_multihead_attn(h, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed, attn_mask, mems, d_model, n_head, d_head, dropout, dropatt, is_training, kernel_initializer,  scope='rel_attn', reuse=None,  )
rel_shift(x,  klen=-1,  )
relative_positional_encoding(qlen, klen, d_model, clamp_len, attn_type, bi_data,  bsz=None, dtype=None,  )
summarize_sequence(summary_type, hidden, d_model, n_head, d_head, dropout, dropatt, input_mask, is_training, initializer,  scope=None, reuse=None, use_proj=True,  )
transformer_xl(inp_k, n_token, n_layer, d_model, n_head, d_head, d_inner, dropout, dropatt, attn_type, bi_data, initializer, is_training,  mem_len=None, inp_q=None, mems=None, same_length=False, clamp_len=-1, untie_r=False, use_tpu=True, input_mask=None, perm_mask=None, seg_id=None, reuse_len=None, ff_activation='relu', target_mapping=None, use_bfloat16=False, scope='transformer',  **kwargs)
two_stream_rel_attn(h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed, attn_mask_h, attn_mask_g, target_mapping, d_model, n_head, d_head, dropout, dropatt, is_training, kernel_initializer,  scope='rel_attn',  )


mlmodels\model_dev\raw\topic-model\utils.py
----------------methods----------------

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
separate_dataset(trainset,  ratio=0.5,  )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_dev\raw\topic-model\xlnet.py
----------------methods----------------
RunConfig.__init__(self, is_training, use_tpu, use_bfloat16, dropout, dropatt,  init='normal', init_range=0.1, init_std=0.02, mem_len=None, reuse_len=None, bi_data=False, clamp_len=-1, same_length=False,  )
XLNetConfig.__init__(self,  FLAGS=None, json_path=None,  )
XLNetConfig.init_from_flags(self, FLAGS,   )
XLNetConfig.init_from_json(self, json_path,   )
XLNetConfig.to_json(self, json_path,   )
XLNetModel.__init__(self, xlnet_config, run_config, input_ids, seg_ids, input_mask,  mems=None, perm_mask=None, target_mapping=None, inp_q=None,  **kwargs)
XLNetModel.get_embedding_table(self,   )
XLNetModel.get_initializer(self,   )
XLNetModel.get_new_memory(self,   )
XLNetModel.get_pooled_out(self, summary_type,  use_summ_proj=True,  )
XLNetModel.get_sequence_output(self,   )

---------------functions---------------
_get_initializer(FLAGS,   )
create_run_config(is_training, is_finetune, FLAGS,   )


mlmodels\model_dev\raw\vectorizer\utils.py
----------------methods----------------

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
separate_dataset(trainset,  ratio=0.5,  )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_flow\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_flow\dev\mlflow_run.py
----------------methods----------------

---------------functions---------------
cli_load_arguments(  )
log_scalar(name, value, step,   )
mlflow_add(args,   )
session_init(args,   )
tfboard_add_weights(step,   )
tfboard_writer_create(  )


mlmodels\model_gluon\fb_prophet.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
fit( model=None, data_pars={}, compute_pars={}, out_pars={},  **kw)
get_dataset(data_pars,   )
get_params( param_pars={},  **kw)
load( load_pars={},  **kw)
metrics_plot(metrics_params,   )
predict( model=None, model_pars=None, sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
save( model=None, session=None, save_pars={},  )
test( data_path="dataset/", pars_choice="test0", config_mode="test",  )


mlmodels\model_gluon\gluonts_model.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)

---------------functions---------------
fit(model,  sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
fit_metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kw)
get_dataset(data_pars,   )
get_params( choice="", data_path="dataset/timeseries/", config_mode="test",  **kw)
load(path,   )
metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kw)
plot_predict(item_metrics,  out_pars=None,  )
plot_prob_forecasts(ypred,  out_pars=None,  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
save(model, path,   )
test(  )
test_single( data_path="dataset/", choice="", config_mode="test",  )


mlmodels\model_gluon\gluon_automl.py
----------------methods----------------
Model.__init__(self,  model_pars=None, compute_pars=None,  )

---------------functions---------------
_config_process(config,   )
get_params( choice="", data_path="dataset/", config_mode="test",  **kw)
path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
test( data_path="dataset/", pars_choice="json",  )


mlmodels\model_gluon\util.py
----------------methods----------------
Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )

---------------functions---------------
_config_process(data_path,  config_mode="test",  )
fit(model,  sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
get_dataset(data_pars,   )
load(path,   )
metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
plot_predict(item_metrics,  out_pars=None,  )
plot_prob_forecasts(ypred,  out_pars=None,  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
save(model, path,   )


mlmodels\model_gluon\util_autogluon.py
----------------methods----------------
Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )

---------------functions---------------
_get_dataset_from_aws(  **kw)
fit(model,  data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
get_dataset(  **kw)
import_data_fromfile(  **kw)
load(path,   )
log( n=0, m=1,  *s)
metrics(model, ypred, ytrue, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
os_package_root_path(filepath,  sublevel=0, path_add="",  )
predict(model, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
save(model, out_pars,   )


mlmodels\model_gluon\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_gluon\raw\gluon_prophet.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
get_params( choice="", data_path="dataset/", config_mode="test",  **kw)
test( data_path="dataset/", choice="",  )


mlmodels\model_gluon\ztest\zNotUse_gluon_deepar.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )

---------------functions---------------
_config_process(data_path,  config_mode="test", choice="test01",  )
fit(modeule, model,  sess=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
get_dataset(data_pars,   )
get_params( choice="", data_path="dataset/timeseries/", config_mode="test",  **kw)
load(path,   )
metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
plot_predict(item_metrics,  out_pars=None,  )
plot_prob_forecasts(ypred,  out_pars=None,  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
save(model, path,   )
test( data_path="dataset/", choice="",  )


mlmodels\model_gluon\ztest\zNo_gluon_ffn.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)

---------------functions---------------
get_params( choice="", data_path="dataset/timeseries/", config_mode="test",  **kw)
test( data_path="dataset/", choice="test01",  )


mlmodels\model_keras\01_deepctr.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)

---------------functions---------------
_config_process(config,   )
_preprocess_criteo(df,   **kw)
_preprocess_movielens(df,   **kw)
config_load(data_path, file_default, config_mode,   )
fit(model,  session=None, compute_pars=None, data_pars=None, out_pars=None,  **kwargs)
get_dataset( data_pars=None,  **kw)
get_params( choice="", data_path="dataset/", config_mode="test",  **kwargs)
metrics(ypred,  ytrue=None, session=None, compute_pars=None, data_pars=None, out_pars=None,  **kwargs)
path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
predict(model,  session=None, compute_pars=None, data_pars=None, out_pars=None,  **kwargs)
reset_model(  )
test( data_path="dataset/", pars_choice=0,  **kwargs)


mlmodels\model_keras\02_cnn.py
----------------methods----------------
Model.__init__(self,  model_pars=None, compute_pars=None, data_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
get_dataset(data_pars,   **kw)
get_params( choice=0, data_path="dataset/",  **kw)
load( load_pars={},  )
log( n=0, m=1,  *s)
metrics(ypred, model,  session=None, model_pars=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
os_package_root_path(filepath,  sublevel=0, path_add="",  )
predict(model,  session=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
save( model=None, session=None, save_pars={},  )
test( data_path="dataset/",  )
test2( data_path="dataset/", out_path="keras/keras.png", reset=True,  )


mlmodels\model_keras\armdn.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
fit( model=None, data_pars={}, compute_pars={}, out_pars={},  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None, model_pars=None,  **kw)
get_dataset(data_pars,   )
get_params( param_pars={},  **kw)
load( load_pars={},  **kw)
metrics_eval(model,  data_pars=None, compute_pars=None, out_pars=None, model_pars=None,  **kw)
metrics_plot(metrics_params,   )
predict( model=None, model_pars=None, sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
reset_model(  )
save( model=None, session=None, save_pars={},  )
test( data_path="dataset/", pars_choice="test0", config_mode="test",  )


mlmodels\model_keras\Autokeras.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None, out_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  )
get_config_file(  )
get_dataset( data_pars=None,  )
get_dataset_imbd(data_pars,   )
get_params( param_pars=None,  **kw)
load(load_pars,   )
predict(model,  session=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
save(model,  session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\charcnn.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset( data_pars=None,  **kw)
get_params( param_pars={},  **kw)
load( load_pars=None,  )
predict(model,  session=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
reset_model(  )
save( model=None, save_pars=None, session=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\charcnn_zhang.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
fit(model,  data_pars={}, compute_pars={}, out_pars={},  **kw)
fit_metrics(model,  data_pars={}, compute_pars={}, out_pars={},  **kw)
get_dataset( data_pars=None,  **kw)
get_params( param_pars={},  **kw)
load( load_pars={},  )
predict(model,  sess=None, data_pars={}, out_pars={}, compute_pars={},  **kw)
reset_model(  )
save( model=None, session=None, save_pars={},  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\keras_gan.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None, out_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  )
get_config_file(  )
get_dataset( data_pars=None,  **kw)
get_params( param_pars=None,  **kw)
load(load_pars,   )
predict(model,  session=None, data_pars=None, compute_pars=None, out_pars=None,  )
save(model,  session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\namentity_crm_bilstm.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)

---------------functions---------------
_preprocess_test(data_pars,   **kw)
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset(data_pars,   **kw)
get_params( param_pars={},  **kw)
load(load_pars,   )
predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
reset_model(  )
save( model=None, session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\namentity_crm_bilstm_dataloader.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset(data_pars,   )
get_params( param_pars={},  **kw)
load(load_pars,   )
predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
reset_model(  )
save( model=None, session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\nbeats.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  session=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset( data_pars=None,  **kw)
get_params( param_pars={},  **kw)
load( load_pars=None,  )
main(  )
predict(model,  session=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
reset_model(  )
save( model=None, save_pars=None, session=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\preprocess.py
----------------methods----------------

---------------functions---------------
_preprocess_criteo(df,   **kw)
_preprocess_movielens(df,   **kw)
_preprocess_none(df,   **kw)
get_dataset(  **kw)
log( n=0, m=1,  *s)
os_package_root_path(filepath,  sublevel=0, path_add="",  )
test( data_path="dataset/", pars_choice=0,  )


mlmodels\model_keras\textcnn.py
----------------methods----------------

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset( data_pars=None,  **kw)
get_params( param_pars={},  **kw)
load( load_pars={},  )
predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
reset_model(  )
save( model=None, session=None, save_pars={},  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\textvae.py
----------------methods----------------
CustomVariationalLayer.__init__(self,   **kwargs)
CustomVariationalLayer.call(self, inputs,   )
CustomVariationalLayer.vae_loss(self, x, x_decoded_mean,   )
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset( data_pars=None,  **kw)
get_params( param_pars={},  **kw)
load( load_pars={},  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
reset_model(  )
save( model=None, session=None, save_pars={},  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_keras\util.py
----------------methods----------------
Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )

---------------functions---------------
_config_process(data_path,  config_mode="test",  )
fit(model,  data_pars=None, model_pars=None, compute_pars=None, out_pars=None, session=None,  **kwargs)
get_dataset(  **kw)
load(path,   )
log( n=0, m=1,  *s)
metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
os_package_root_path(filepath,  sublevel=0, path_add="",  )
predict(model, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
save(model, path,   )


mlmodels\model_keras\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\no_03_textcnn.py
----------------methods----------------

---------------functions---------------
fit(model, Xtrain, ytrain,  compute_pars=None,  **kw)
get_params( choice="", data_path="./dataset/", config_mode="test",  **kw)
get_pre_train_word2vec(model, index2word, vocab_size,   )
load(path,   )
log( n=0, m=1,  *s)
metrics(ytrue, ypred,  data_pars=None, out_pars=None,  **kw)
os_module_path(  )
path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
predict(model, Xtest, ytest,  data_pars=None, out_pars=None, compute_pars=None,  **kw)
reset_model(  )
save(model, path,   )
test( data_path="dataset/", pars_choice="json", reset=True,  )
test2( data_path="dataset/", pars_choice="json", reset=True,  )


mlmodels\model_keras\raw\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\char_cnn\data_utils.py
----------------methods----------------
Data.__init__(self, data_source,  alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,  )
Data.fill_to_length_input(self, s,   )
Data.get_all_data(self,   )
Data.get_all_data_npz(self,   )
Data.load_data(self,   )
Data.str_to_indexes(self, s,   )

---------------functions---------------


mlmodels\model_keras\raw\char_cnn\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\char_cnn\models\char_cnn_kim.py
----------------methods----------------
CharCNNKim.__init__(self, input_size, alphabet_size, embedding_size, conv_layers, fully_connected_layers, num_of_classes, dropout_p,  optimizer='adam', loss='categorical_crossentropy',  )
CharCNNKim._build_model(self,   )
CharCNNKim.test(self, testing_inputs, testing_labels, batch_size,   )
CharCNNKim.train(self, training_inputs, training_labels, validation_inputs, validation_labels, epochs, batch_size,  checkpoint_every=100,  )

---------------functions---------------


mlmodels\model_keras\raw\char_cnn\models\char_cnn_zhang.py
----------------methods----------------
CharCNNZhang.__init__(self, input_size, alphabet_size, embedding_size, conv_layers, fully_connected_layers, num_of_classes, threshold, dropout_p,  optimizer='adam', loss='categorical_crossentropy',  )
CharCNNZhang._build_model(self,   )
CharCNNZhang.test(self, testing_inputs, testing_labels, batch_size,   )
CharCNNZhang.train(self, training_inputs, training_labels, validation_inputs, validation_labels, epochs, batch_size,  checkpoint_every=100,  )

---------------functions---------------


mlmodels\model_keras\raw\char_cnn\models\char_tcn.py
----------------methods----------------
CharTCN.__init__(self, input_size, alphabet_size, embedding_size, conv_layers, fully_connected_layers, num_of_classes, threshold, dropout_p,  optimizer='adam', loss='categorical_crossentropy',  )
CharTCN._build_model(self,   )
CharTCN.test(self, testing_inputs, testing_labels, batch_size,   )
CharTCN.train(self, training_inputs, training_labels, validation_inputs, validation_labels, epochs, batch_size,  checkpoint_every=100,  )

---------------functions---------------


mlmodels\model_keras\raw\char_cnn\models\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\examples\run_classification_criteo.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\examples\run_classification_criteo_hash.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\examples\run_classification_criteo_multi_gpu.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\examples\run_dien.py
----------------methods----------------

---------------functions---------------
get_xy_fd( use_neg=False, hash_flag=False,  )


mlmodels\model_keras\raw\examples\run_din.py
----------------methods----------------

---------------functions---------------
get_xy_fd(  )


mlmodels\model_keras\raw\examples\run_dsin.py
----------------methods----------------

---------------functions---------------
get_xy_fd( hash_flag=False,  )


mlmodels\model_keras\raw\examples\run_multivalue_movielens.py
----------------methods----------------

---------------functions---------------
split(x,   )


mlmodels\model_keras\raw\examples\run_multivalue_movielens_hash.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\examples\run_regression_movielens.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\FastText\fast_text.py
----------------methods----------------
FastText.__init__(self, maxlen, max_features, embedding_dims,  class_num=1, last_activation='sigmoid',  )
FastText.get_model(self,   )

---------------functions---------------


mlmodels\model_keras\raw\FastText\main.py
----------------methods----------------

---------------functions---------------
add_ngram(sequences, token_indice,  ngram_range=2,  )
create_ngram_set(input_list,  ngram_value=2,  )


mlmodels\model_keras\raw\graph_cnn_text\build_graph.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\graph_cnn_text\inits.py
----------------methods----------------

---------------functions---------------
glorot(shape,  name=None,  )
ones(shape,  name=None,  )
uniform(shape,  scale=0.05, name=None,  )
weight_variable_glorot(input_dim, output_dim,  name="",  )
zeros(shape,  name=None,  )


mlmodels\model_keras\raw\graph_cnn_text\layers.py
----------------methods----------------
Dense.__init__(self, input_dim, output_dim, placeholders,  dropout=0., sparse_inputs=False, act=tf.nn.relu, bias=False, featureless=False,  **kwargs)
Dense._call(self, inputs,   )
GraphConvolution.__init__(self, input_dim, output_dim, placeholders,  dropout=0., sparse_inputs=False, act=tf.nn.relu, bias=False, featureless=False,  **kwargs)
GraphConvolution._call(self, inputs,   )
Layer.__call__(self, inputs,   )
Layer.__init__(self,  edge_type=(), num_types=-1,  **kwargs)
Layer._call(self, inputs,   )
Layer._log_vars(self,   )

---------------functions---------------
dot(x, y,  sparse=False,  )
dropout_sparse(x, keep_prob, num_nonzero_elems,   )
get_layer_uid( layer_name='',  )
sparse_dropout(x, keep_prob, noise_shape,   )


mlmodels\model_keras\raw\graph_cnn_text\metrics.py
----------------methods----------------

---------------functions---------------
masked_accuracy(preds, labels, mask,   )
masked_softmax_cross_entropy(preds, labels, mask,   )


mlmodels\model_keras\raw\graph_cnn_text\models.py
----------------methods----------------
GCN.__init__(self, placeholders, input_dim,   **kwargs)
GCN._accuracy(self,   )
GCN._build(self,   )
GCN._loss(self,   )
GCN.predict(self,   )
Model.__init__(self,   **kwargs)
Model._accuracy(self,   )
Model._build(self,   )
Model._loss(self,   )
Model.build(self,   )
Model.load(self,  sess=None,  )
Model.predict(self,   )
Model.save(self,  sess=None,  )
RGCN.__init__(self, placeholders, num_feat, nonzero_feat, edge_types,   **kwargs)
RGCN._accuracy(self,   )
RGCN._build(self,   )
RGCN._loss(self,   )

---------------functions---------------


mlmodels\model_keras\raw\graph_cnn_text\remove_words.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\graph_cnn_text\train.py
----------------methods----------------

---------------functions---------------
evaluate(features, support, labels, mask, placeholders,   )


mlmodels\model_keras\raw\graph_cnn_text\tsne.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\graph_cnn_text\utils.py
----------------methods----------------

---------------functions---------------
build_feed_dict(labels, labels_mask, adj, edge_types, feat, placeholders,   )
chebyshev_polynomials(adj, k,   )
clean_str(string,   )
construct_feed_dict(features, support, labels, labels_mask, placeholders,   )
loadWord2Vec(filename,   )
load_corpus(dataset_str,   )
load_corpus_kg(dataset_str,   )
load_corpus_multimodal(dataset_str,   )
load_data(dataset_str,   )
normalize_adj(adj,  symmetric=True,  )
parse_index_file(filename,   )
preprocess_adj(adj,   )
preprocess_features(features,   )
preprocess_graph(adj,  symmetric=True,  )
read_triples(file_path,   )
sample_mask(idx, l,   )
sparse_to_tuple(sparse_mx,   )
synonimize(word,  pos=None,  )
word_synonyms(word,   )
wordnet_defs(  )
wordnet_id_num_dict(  )
wordnet_id_synset_dict(  )


mlmodels\model_keras\raw\graph_cnn_text\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\HAN\attention.py
----------------methods----------------
Attention.__init__(self, step_dim,  W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True,  **kwargs)
Attention.build(self, input_shape,   )
Attention.call(self, x,  mask=None,  )
Attention.compute_mask(self, input,  input_mask=None,  )
Attention.compute_output_shape(self, input_shape,   )

---------------functions---------------


mlmodels\model_keras\raw\HAN\han.py
----------------methods----------------
HAN.__init__(self, maxlen_sentence, maxlen_word, max_features, embedding_dims,  class_num=1, last_activation='sigmoid',  )
HAN.get_model(self,   )

---------------functions---------------


mlmodels\model_keras\raw\HAN\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\aae\aae.py
----------------methods----------------
AdversarialAutoencoder.__init__(self,   )
AdversarialAutoencoder.build_decoder(self,   )
AdversarialAutoencoder.build_discriminator(self,   )
AdversarialAutoencoder.build_encoder(self,   )
AdversarialAutoencoder.sample_images(self, epoch,   )
AdversarialAutoencoder.save_model(self,   )
AdversarialAutoencoder.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\acgan\acgan.py
----------------methods----------------
ACGAN.__init__(self,   )
ACGAN.build_discriminator(self,   )
ACGAN.build_generator(self,   )
ACGAN.sample_images(self, epoch,   )
ACGAN.save_model(self,   )
ACGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\bgan\bgan.py
----------------methods----------------
BGAN.__init__(self,   )
BGAN.boundary_loss(self, y_true, y_pred,   )
BGAN.build_discriminator(self,   )
BGAN.build_generator(self,   )
BGAN.sample_images(self, epoch,   )
BGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\bigan\bigan.py
----------------methods----------------
BIGAN.__init__(self,   )
BIGAN.build_discriminator(self,   )
BIGAN.build_encoder(self,   )
BIGAN.build_generator(self,   )
BIGAN.sample_interval(self, epoch,   )
BIGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\ccgan\ccgan.py
----------------methods----------------
CCGAN.__init__(self,   )
CCGAN.build_discriminator(self,   )
CCGAN.build_generator(self,   )
CCGAN.mask_randomly(self, imgs,   )
CCGAN.sample_images(self, epoch, imgs,   )
CCGAN.save_model(self,   )
CCGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\cgan\cgan.py
----------------methods----------------
CGAN.__init__(self,   )
CGAN.build_discriminator(self,   )
CGAN.build_generator(self,   )
CGAN.sample_images(self, epoch,   )
CGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\cogan\cogan.py
----------------methods----------------
COGAN.__init__(self,   )
COGAN.build_discriminators(self,   )
COGAN.build_generators(self,   )
COGAN.sample_images(self, epoch,   )
COGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\context_encoder\context_encoder.py
----------------methods----------------
ContextEncoder.__init__(self,   )
ContextEncoder.build_discriminator(self,   )
ContextEncoder.build_generator(self,   )
ContextEncoder.mask_randomly(self, imgs,   )
ContextEncoder.sample_images(self, epoch, imgs,   )
ContextEncoder.save_model(self,   )
ContextEncoder.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\cyclegan\cyclegan.py
----------------methods----------------
CycleGAN.__init__(self,   )
CycleGAN.build_discriminator(self,   )
CycleGAN.build_generator(self,   )
CycleGAN.sample_images(self, epoch, batch_i,   )
CycleGAN.train(self, epochs,  batch_size=1, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\cyclegan\data_loader.py
----------------methods----------------
DataLoader.__init__(self, dataset_name,  img_res=(128, 128),  )
DataLoader.imread(self, path,   )
DataLoader.load_batch(self,  batch_size=1, is_testing=False,  )
DataLoader.load_data(self, domain,  batch_size=1, is_testing=False,  )
DataLoader.load_img(self, path,   )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\dcgan\dcgan.py
----------------methods----------------
DCGAN.__init__(self,   )
DCGAN.build_discriminator(self,   )
DCGAN.build_generator(self,   )
DCGAN.save_imgs(self, epoch,   )
DCGAN.train(self, epochs,  batch_size=128, save_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\discogan\data_loader.py
----------------methods----------------
DataLoader.__init__(self, dataset_name,  img_res=(128, 128),  )
DataLoader.imread(self, path,   )
DataLoader.load_batch(self,  batch_size=1, is_testing=False,  )
DataLoader.load_data(self,  batch_size=1, is_testing=False,  )
DataLoader.load_img(self, path,   )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\discogan\discogan.py
----------------methods----------------
DiscoGAN.__init__(self,   )
DiscoGAN.build_discriminator(self,   )
DiscoGAN.build_generator(self,   )
DiscoGAN.sample_images(self, epoch, batch_i,   )
DiscoGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\dualgan\dualgan.py
----------------methods----------------
DUALGAN.__init__(self,   )
DUALGAN.build_discriminator(self,   )
DUALGAN.build_generator(self,   )
DUALGAN.sample_generator_input(self, X, batch_size,   )
DUALGAN.save_imgs(self, epoch, X_A, X_B,   )
DUALGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )
DUALGAN.wasserstein_loss(self, y_true, y_pred,   )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\gan\gan.py
----------------methods----------------
GAN.__init__(self,   )
GAN.build_discriminator(self,   )
GAN.build_generator(self,   )
GAN.sample_images(self, epoch,   )
GAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\infogan\infogan.py
----------------methods----------------
INFOGAN.__init__(self,   )
INFOGAN.build_disk_and_q_net(self,   )
INFOGAN.build_generator(self,   )
INFOGAN.mutual_info_loss(self, c, c_given_x,   )
INFOGAN.sample_generator_input(self, batch_size,   )
INFOGAN.sample_images(self, epoch,   )
INFOGAN.save_model(self,   )
INFOGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\lsgan\lsgan.py
----------------methods----------------
LSGAN.__init__(self,   )
LSGAN.build_discriminator(self,   )
LSGAN.build_generator(self,   )
LSGAN.sample_images(self, epoch,   )
LSGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\pix2pix\data_loader.py
----------------methods----------------
DataLoader.__init__(self, dataset_name,  img_res=(128, 128),  )
DataLoader.imread(self, path,   )
DataLoader.load_batch(self,  batch_size=1, is_testing=False,  )
DataLoader.load_data(self,  batch_size=1, is_testing=False,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\pix2pix\pix2pix.py
----------------methods----------------
Pix2Pix.__init__(self,   )
Pix2Pix.build_discriminator(self,   )
Pix2Pix.build_generator(self,   )
Pix2Pix.sample_images(self, epoch, batch_i,   )
Pix2Pix.train(self, epochs,  batch_size=1, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\pixelda\data_loader.py
----------------methods----------------
DataLoader.__init__(self,  img_res=(128, 128),  )
DataLoader.load_data(self, domain,  batch_size=1,  )
DataLoader.normalize(self, images,   )
DataLoader.setup_mnist(self, img_res,   )
DataLoader.setup_mnistm(self, img_res,   )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\pixelda\pixelda.py
----------------methods----------------
PixelDA.__init__(self,   )
PixelDA.build_classifier(self,   )
PixelDA.build_discriminator(self,   )
PixelDA.build_generator(self,   )
PixelDA.sample_images(self, epoch,   )
PixelDA.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\pixelda\test.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\sgan\sgan.py
----------------methods----------------
SGAN.__init__(self,   )
SGAN.build_discriminator(self,   )
SGAN.build_generator(self,   )
SGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\srgan\data_loader.py
----------------methods----------------
DataLoader.__init__(self, dataset_name,  img_res=(128, 128),  )
DataLoader.imread(self, path,   )
DataLoader.load_data(self,  batch_size=1, is_testing=False,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\srgan\srgan.py
----------------methods----------------
SRGAN.__init__(self,   )
SRGAN.build_discriminator(self,   )
SRGAN.build_generator(self,   )
SRGAN.build_vgg(self,   )
SRGAN.sample_images(self, epoch,   )
SRGAN.train(self, epochs,  batch_size=1, sample_interval=50,  )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\wgan\wgan.py
----------------methods----------------
WGAN.__init__(self,   )
WGAN.build_critic(self,   )
WGAN.build_generator(self,   )
WGAN.sample_images(self, epoch,   )
WGAN.train(self, epochs,  batch_size=128, sample_interval=50,  )
WGAN.wasserstein_loss(self, y_true, y_pred,   )

---------------functions---------------


mlmodels\model_keras\raw\keras_gan\wgan_gp\wgan_gp.py
----------------methods----------------
RandomWeightedAverage._merge_function(self, inputs,   )
WGANGP.__init__(self,   )
WGANGP.build_critic(self,   )
WGANGP.build_generator(self,   )
WGANGP.gradient_penalty_loss(self, y_true, y_pred, averaged_samples,   )
WGANGP.sample_images(self, epoch,   )
WGANGP.train(self, epochs, batch_size,  sample_interval=50,  )
WGANGP.wasserstein_loss(self, y_true, y_pred,   )

---------------functions---------------


mlmodels\model_keras\raw\nbeats_keras\model.py
----------------methods----------------
NBeatsNet.__getattr__(self, name,   )
NBeatsNet.__init__(self,  input_dim=1, exo_dim=0, backcast_length=10, forecast_length=2, stack_types=(TREND_BLOCK, SEASONALITY_BLOCK), nb_blocks_per_stack=3, thetas_dim=(4, 8), share_weights_in_stack=False, hidden_layer_units=256, nb_harmonics=None,  )
NBeatsNet._r(self, layer_with_weights, stack_id,   )
NBeatsNet.compile_model(self, loss, learning_rate,   )
NBeatsNet.create_block(self, x, e, stack_id, block_id, stack_type, nb_poly,   )
NBeatsNet.has_exog(self,   )
NBeatsNet.load(filepath,  custom_objects=None, compile=True,  )

---------------functions---------------
linear_space(backcast_length, forecast_length,  fwd_looking=True,  )
seasonality_model(thetas, backcast_length, forecast_length, is_forecast,   )
trend_model(thetas, backcast_length, forecast_length, is_forecast,   )


mlmodels\model_keras\raw\nbeats_keras\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\RCNN\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\RCNN\rcnn.py
----------------methods----------------
RCNN.__init__(self, maxlen, max_features, embedding_dims,  class_num=1, last_activation='sigmoid',  )
RCNN.get_model(self,   )

---------------functions---------------


mlmodels\model_keras\raw\RCNNVariant\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\RCNNVariant\rcnn_variant.py
----------------methods----------------
RCNNVariant.__init__(self, maxlen, max_features, embedding_dims,  class_num=1, last_activation='sigmoid',  )
RCNNVariant.get_model(self,   )

---------------functions---------------


mlmodels\model_keras\raw\TextAttBiRNN\attention.py
----------------methods----------------
Attention.__init__(self, step_dim,  W_regularizer=None, b_regularizer=None, W_constraint=None, b_constraint=None, bias=True,  **kwargs)
Attention.build(self, input_shape,   )
Attention.call(self, x,  mask=None,  )
Attention.compute_mask(self, input,  input_mask=None,  )
Attention.compute_output_shape(self, input_shape,   )

---------------functions---------------


mlmodels\model_keras\raw\TextAttBiRNN\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\TextAttBiRNN\text_att_birnn.py
----------------methods----------------
TextAttBiRNN.__init__(self, maxlen, max_features, embedding_dims,  class_num=1, last_activation='sigmoid',  )
TextAttBiRNN.get_model(self,   )

---------------functions---------------


mlmodels\model_keras\raw\TextBiRNN\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\TextBiRNN\text_birnn.py
----------------methods----------------
TextBiRNN.__init__(self, maxlen, max_features, embedding_dims,  class_num=1, last_activation='sigmoid',  )
TextBiRNN.get_model(self,   )

---------------functions---------------


mlmodels\model_keras\raw\textcnn_\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\textcnn_\text_cnn.py
----------------methods----------------
TextCNN.__init__(self, maxlen, max_features, embedding_dims,  class_num=1, last_activation='softmax',  )
TextCNN.get_model(self,   )

---------------functions---------------


mlmodels\model_keras\raw\textcnn_\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\TextRNN\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\TextRNN\text_rnn.py
----------------methods----------------
TextRNN.__init__(self, maxlen, max_features, embedding_dims,  class_num=1, last_activation='sigmoid',  )
TextRNN.get_model(self,   )

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\bow.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\build_corpus.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\build_graph.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\doc2vec.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\inits.py
----------------methods----------------

---------------functions---------------
glorot(shape,  name=None,  )
ones(shape,  name=None,  )
uniform(shape,  scale=0.05, name=None,  )
zeros(shape,  name=None,  )


mlmodels\model_keras\raw\text_graph_cnn2\layers.py
----------------methods----------------
Dense.__init__(self, input_dim, output_dim, placeholders,  dropout=0., sparse_inputs=False, act=tf.nn.relu, bias=False, featureless=False,  **kwargs)
Dense._call(self, inputs,   )
GraphConvolution.__init__(self, input_dim, output_dim, placeholders,  dropout=0., sparse_inputs=False, act=tf.nn.relu, bias=False, featureless=False,  **kwargs)
GraphConvolution._call(self, inputs,   )
Layer.__call__(self, inputs,   )
Layer.__init__(self,   **kwargs)
Layer._call(self, inputs,   )
Layer._log_vars(self,   )

---------------functions---------------
dot(x, y,  sparse=False,  )
get_layer_uid( layer_name='',  )
sparse_dropout(x, keep_prob, noise_shape,   )


mlmodels\model_keras\raw\text_graph_cnn2\metrics.py
----------------methods----------------

---------------functions---------------
masked_accuracy(preds, labels, mask,   )
masked_softmax_cross_entropy(preds, labels, mask,   )


mlmodels\model_keras\raw\text_graph_cnn2\models.py
----------------methods----------------
GCN.__init__(self, placeholders, input_dim,   **kwargs)
GCN._accuracy(self,   )
GCN._build(self,   )
GCN._loss(self,   )
GCN.predict(self,   )
MLP.__init__(self, placeholders, input_dim,   **kwargs)
MLP._accuracy(self,   )
MLP._build(self,   )
MLP._loss(self,   )
MLP.predict(self,   )
Model.__init__(self,   **kwargs)
Model._accuracy(self,   )
Model._build(self,   )
Model._loss(self,   )
Model.build(self,   )
Model.load(self,  sess=None,  )
Model.predict(self,   )
Model.save(self,  sess=None,  )

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\plot_dim.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\plot_prop.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\plot_window.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\prepare_data.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\remove_words.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\train.py
----------------methods----------------

---------------functions---------------
evaluate(features, support, labels, mask, placeholders,   )


mlmodels\model_keras\raw\text_graph_cnn2\utils.py
----------------methods----------------

---------------functions---------------
chebyshev_polynomials(adj, k,   )
clean_str(string,   )
construct_feed_dict(features, support, labels, labels_mask, placeholders,   )
loadWord2Vec(filename,   )
load_corpus(dataset_str,   )
load_data(dataset_str,   )
normalize_adj(adj,   )
parse_index_file(filename,   )
preprocess_adj(adj,   )
preprocess_features(features,   )
sample_mask(idx, l,   )
sparse_to_tuple(sparse_mx,   )


mlmodels\model_keras\raw\text_graph_cnn2\visualize.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\visualize_words.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\wordnet.py
----------------methods----------------

---------------functions---------------


mlmodels\model_keras\raw\text_graph_cnn2\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\dev\LambdaRank.py
----------------methods----------------
LambdaRank.__init__(self, net_structures,  leaky_relu=False, sigma=1.0, double_precision=False,  )
LambdaRank.dump_param(self,   )
LambdaRank.forward(self, input1,   )

---------------functions---------------
train( start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam", leaky_relu=False, ndcg_gain_in_train="exp2", sigma=1.0, double_precision=False, standardize=False, small_dataset=False, debug=False,  )


mlmodels\model_rank\dev\load_mslr.py
----------------methods----------------
DataLoader.__init__(self, path,   )
DataLoader._load_mslr(self,   )
DataLoader._parse_feature_and_label(self, df,   )
DataLoader.apply_scaler(self, scaler,   )
DataLoader.generate_batch_per_query(self,  df=None,  )
DataLoader.generate_query_batch(self, df,  batchsize=100000,  )
DataLoader.generate_query_pair_batch(self,  df=None, batchsize=2000,  )
DataLoader.generate_query_pairs(self, df, qid,   )
DataLoader.get_num_pairs(self,   )
DataLoader.get_num_sessions(self,   )
DataLoader.load(self,   )
DataLoader.train_scaler_and_transform(self,   )

---------------functions---------------
get_time(  )


mlmodels\model_rank\dev\metrics.py
----------------methods----------------
DCG.__init__(self,  k=10, gain_type='exp2',  )
DCG._get_discount(self, k,   )
DCG._get_gain(self, targets,   )
DCG._make_discount(n,   )
DCG.evaluate(self, targets,   )
NDCG.__init__(self,  k=10, gain_type='exp2',  )
NDCG.evaluate(self, targets,   )
NDCG.maxDCG(self, targets,   )

---------------functions---------------


mlmodels\model_rank\dev\RankNet.py
----------------methods----------------
RankNet.__init__(self, net_structures,  double_precision=False,  )
RankNet.dump_param(self,   )
RankNet.forward(self, input1,   )
RankNetPairs.__init__(self, net_structures,  double_precision=False,  )
RankNetPairs.forward(self, input1, input2,   )

---------------functions---------------
baseline_pairwise_training_loop(epoch, net, loss_func, optimizer, train_loader,  batch_size=100000, precision=torch.float32, device="cpu", debug=False,  )
eval_model(inference_model, device, df_valid, valid_loader,   )
factorized_training_loop(epoch, net, loss_func, optimizer, train_loader,  batch_size=200, sigma=1.0, training_algo=SUM_SESSION, precision=torch.float32, device="cpu", debug=False,  )
get_train_inference_net(train_algo, num_features, start_epoch, double_precision,   )
load_from_ckpt(ckpt_file, epoch, model,   )
train_rank_net( start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam", train_algo=SUM_SESSION, double_precision=False, standardize=False, small_dataset=False, debug=False,  )


mlmodels\model_rank\dev\utils.py
----------------methods----------------

---------------functions---------------
eval_cross_entropy_loss(model, device, loader,  phase="Eval", sigma=1.0,  )
eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, batch_size, k_list,  phase="Eval",  )
get_args_parser(  )
get_ckptdir(net_name, net_structure,  sigma=None,  )
get_device(  )
init_weights(m,   )
load_train_vali_data(data_fold,  small_dataset=False,  )
save_to_ckpt(ckpt_file, epoch, model, optimizer, lr_scheduler,   )
str2bool(v,   )


mlmodels\model_rank\raw\irgan_tf\item_recommendation\cf_dns.py
----------------methods----------------

---------------functions---------------
dcg_at_k(r, k,   )
generate_dns(sess, model, filename,   )
generate_uniform(filename,   )
main(  )
ndcg_at_k(r, k,   )
simple_test(sess, model,   )
simple_test_one_user(x,   )


mlmodels\model_rank\raw\irgan_tf\item_recommendation\cf_gan.py
----------------methods----------------

---------------functions---------------
dcg_at_k(r, k,   )
generate_for_d(sess, model, filename,   )
main(  )
ndcg_at_k(r, k,   )
simple_test(sess, model,   )
simple_test_one_user(x,   )


mlmodels\model_rank\raw\irgan_tf\item_recommendation\dis_model.py
----------------methods----------------
DIS.__init__(self, itemNum, userNum, emb_dim, lamda,  param=None, initdelta=0.05, learning_rate=0.05,  )
DIS.save_model(self, sess, filename,   )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\item_recommendation\dis_model_dns.py
----------------methods----------------
DIS.__init__(self, itemNum, userNum, emb_dim, lamda,  param=None, initdelta=0.05, learning_rate=0.05,  )
DIS.save_model(self, sess, filename,   )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\item_recommendation\gen_model.py
----------------methods----------------
GEN.__init__(self, itemNum, userNum, emb_dim, lamda,  param=None, initdelta=0.05, learning_rate=0.05,  )
GEN.save_model(self, sess, filename,   )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\item_recommendation\utils.py
----------------methods----------------

---------------functions---------------
F1(pre, rec,   )
average_precision(r,   )
dcg_at_k(r, k,  method=1,  )
file_len(fname,   )
get_batch_data(file, index, size,   )
mean_average_precision(rs,   )
ndcg_at_k(r, k,  method=1,  )
precision_at_k(r, k,   )
recall_at_k(r, k, all_pos_num,   )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\dis_model_pairwise_nn.py
----------------methods----------------
DIS.__init__(self, feature_size, hidden_size, weight_decay, learning_rate,  loss='log', param=None,  )
DIS.save_model(self, sess, filename,   )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\gen_model_nn.py
----------------methods----------------
GEN.__init__(self, feature_size, hidden_size, weight_decay, learning_rate,  param=None,  )
GEN.save_model(self, sess, filename,   )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\ltr_gan_d_nn_g_nn.py
----------------methods----------------

---------------functions---------------
generate_for_d(sess, model, filename,   )
main(  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\run.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\utils.py
----------------methods----------------

---------------functions---------------
file_len(fname,   )
get_batch_data(file, index, size,   )
get_query_pos(file,   )
load_all_query_url_feature(file, feature_size,   )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\eval\map.py
----------------methods----------------

---------------functions---------------
MAP(sess, model, query_pos_test, query_pos_train, query_url_feature,   )
MAP_user(sess, model, query_pos_test, query_pos_train, query_url_feature,   )
average_precision(r,   )
precision_at_k(r, k,   )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\eval\mrr.py
----------------methods----------------

---------------functions---------------
MRR(sess, model, query_pos_test, query_pos_train, query_url_feature,   )
MRR_user(sess, model, query_pos_test, query_pos_train, query_url_feature,   )
cal_mrr(r,   )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\eval\ndcg.py
----------------methods----------------

---------------functions---------------
ndcg_at_k(sess, model, query_pos_test, query_pos_train, query_url_feature,  k=5,  )
ndcg_at_k_user(sess, model, query_pos_test, query_pos_train, query_url_feature,  k=5,  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\eval\precision.py
----------------methods----------------

---------------functions---------------
precision_at_k(sess, model, query_pos_test, query_pos_train, query_url_feature,  k=5,  )
precision_at_k_user(sess, model, query_pos_test, query_pos_train, query_url_feature,  k=5,  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pairwise\eval\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\dis_model_pairwise_nn.py
----------------methods----------------
DIS.__init__(self, feature_size, hidden_size, weight_decay, learning_rate,  loss='log', param=None,  )
DIS.save_model(self, sess, filename,   )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\dis_model_pointwise_nn.py
----------------methods----------------
DIS.__init__(self, feature_size, hidden_size, weight_decay, learning_rate,  param=None,  )
DIS.save_model(self, sess, filename,   )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\gen_model_nn.py
----------------methods----------------
GEN.__init__(self, feature_size, hidden_size, weight_decay, learning_rate,  temperature=1.0, param=None,  )
GEN.save_model(self, sess, filename,   )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\ltr_dns_nn.py
----------------methods----------------

---------------functions---------------
generate_dns(sess, model, filename,   )
main(  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\ltr_gan_d_nn_g_nn.py
----------------methods----------------

---------------functions---------------
generate_for_d(sess, model, filename,   )
main(  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\ltr_mle_nn.py
----------------methods----------------

---------------functions---------------
generate_uniform(filename,   )
main(  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\ltr_rns_nn.py
----------------methods----------------

---------------functions---------------
generate_uniform(filename,   )
main(  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\run.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\utils.py
----------------methods----------------

---------------functions---------------
file_len(fname,   )
get_batch_data(file, index, size,   )
get_query_pos(file,   )
load_all_query_url_feature(file, feature_size,   )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\eval\map.py
----------------methods----------------

---------------functions---------------
MAP(sess, model, query_pos_test, query_pos_train, query_url_feature,   )
MAP_user(sess, model, query_pos_test, query_pos_train, query_url_feature,   )
average_precision(r,   )
precision_at_k(r, k,   )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\eval\mrr.py
----------------methods----------------

---------------functions---------------
MRR(sess, model, query_pos_test, query_pos_train, query_url_feature,   )
MRR_user(sess, model, query_pos_test, query_pos_train, query_url_feature,   )
cal_mrr(r,   )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\eval\ndcg.py
----------------methods----------------

---------------functions---------------
ndcg_at_k(sess, model, query_pos_test, query_pos_train, query_url_feature,  k=5,  )
ndcg_at_k_user(sess, model, query_pos_test, query_pos_train, query_url_feature,  k=5,  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\eval\precision.py
----------------methods----------------

---------------functions---------------
precision_at_k(sess, model, query_pos_test, query_pos_train, query_url_feature,  k=5,  )
precision_at_k_user(sess, model, query_pos_test, query_pos_train, query_url_feature,  k=5,  )


mlmodels\model_rank\raw\irgan_tf\ltr-gan\ltr-gan-pointwise\eval\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\Question-Answer\baseline.py
----------------methods----------------

---------------functions---------------
dev_step(sess, cnn, testList,  dev_size=100,  )
evaluation(sess, model, log,  num_epochs=0,  )
generate_dns_pair(sess, model,   )
generate_uniform_pair(  )
log_time_delta(func,   )
main(  )


mlmodels\model_rank\raw\irgan_tf\Question-Answer\dataPrepare.py
----------------methods----------------

---------------functions---------------
convert2TSV(rawFilename,   )
convertAll( subset_size=0,  )
format_file( filename="insurance_dev.tsv", subset_size=1800,  )
load(file_name,   )
parseTrain(  )


mlmodels\model_rank\raw\irgan_tf\Question-Answer\Discriminator.py
----------------methods----------------
Discriminator.__init__(self, sequence_length, batch_size, vocab_size, embedding_size, filter_sizes, num_filters,  dropout_keep_prob=1.0, l2_reg_lambda=0.0, learning_rate=1e-2, paras=None, embeddings=None, loss="pair", trainable=True,  )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\Question-Answer\Generator.py
----------------methods----------------
Generator.__init__(self, sequence_length, batch_size, vocab_size, embedding_size, filter_sizes, num_filters,  dropout_keep_prob=1.0, l2_reg_lambda=0.0, paras=None, learning_rate=1e-2, embeddings=None, loss="pair", trainable=True,  )

---------------functions---------------


mlmodels\model_rank\raw\irgan_tf\Question-Answer\insurance_qa_data_helpers.py
----------------methods----------------

---------------functions---------------
batch_iter(data, batch_size,  num_epochs=1, shuffle=False,  )
build_vocab(  )
encode_sent(vocab, string, size,   )
loadCandidateSamples(q, a, candidates, vocab,   )
loadTestSet(filename,   )
load_train_random(vocab, alist, raw, size,   )
load_val_batch(testList, vocab, index, batch,   )
load_vectors( vocab=None,  )
main(  )
read_alist(  )
read_raw(  )


mlmodels\model_rank\raw\irgan_tf\Question-Answer\QACNN.py
----------------methods----------------
QACNN.__init__(self, sequence_length, batch_size, vocab_size, embedding_size, filter_sizes, num_filters,  dropout_keep_prob=1.0, l2_reg_lambda=0.0, paras=None, learning_rate=1e-2, embeddings=None, loss="pair", trainable=True,  )
QACNN.cosine(self, q, a,   )
QACNN.getRepresentation(self, sentence,   )
QACNN.save_model(self, sess,  precision_current=0,  )

---------------functions---------------


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\Eval.py
----------------methods----------------

---------------functions---------------
DCG(topN, rates,   )
EvalQuery(pHash,   )
FoldMap(hsResult,   )
FoldMeanNdcg(hsResult,   )
FoldNdcg(hsResult,   )
FoldPrecAtN(hsResult,   )
MAP(rates,   )
NDCG(topN, rates,   )
PrecisionAtN(topN, rates,   )
ReadInputFiles(fnFeature, fnPred,   )
f(x,   )


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\ListMLE_conv2d.py
----------------methods----------------
Net.__init__(self, m,   )
Net.eval(self, input_this, output_this, outputpath_this, trainpath_this, evalpath_this,   )
Net.forward(self, input,   )
Net.print_param(self,   )
Net.save_scores(self, scores_test, output_test, test_output_path,   )
Net.seqMLELoss(self, scores, output,   )

---------------functions---------------
get_MAP(split, iter,   )
get_index(split, method,   )


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\plot_ground_random_level_1.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\plot_ListMLE.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\plot_ndcg_log.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\plot_random_level.py
----------------methods----------------

---------------functions---------------


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\RankSVM.py
----------------methods----------------
Net.__init__(self, m,   )
Net.eval(self, input_this, output_this, outputpath_this, trainpath_this, evalpath_this,   )
Net.forward(self, input,   )
Net.pairwiseHinge(self, scores_pos, scores_neg, output_pos, output_neg,   )
Net.print_param(self,   )
Net.save_scores(self, scores_test, output_test, test_output_path,   )

---------------functions---------------


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\reinforce.py
----------------methods----------------
Net.__init__(self, m,   )
Net.eval(self, input_this, output_this, outputpath_this, trainpath_this, evalpath_this, neg_log_sum_loss,   )
Net.forward(self, input,   )
Net.print_param(self,   )
Net.save_scores(self, scores_test, output_test, test_output_path,   )
Net.seqMLELoss(self, scores, output,   )

---------------functions---------------
MDP_for_gradient(model, input_unsorted, output_unsorted,   )
eval_reward(pred_scores, groundtruth_scores, predpath_this, trainpath_this, evalpath_this,   )
sample_episode(model, input_unsorted, output_unsorted,   )


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\reinforce_cartpole_example.py
----------------methods----------------
Policy.__init__(self,   )
Policy.forward(self, x,   )

---------------functions---------------
finish_episode(  )
select_action(state,   )


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\utils.py
----------------methods----------------

---------------functions---------------
load_data_ListMLE(path,  normalized=False,  )
load_data_RankSVM(path,  normalized=False,  )
normalize_by_column(matrix,   )


mlmodels\model_rank\raw\rl_ltr\loss_vs_metric\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_sklearn\model_lightgbm.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset( data_pars=None,  **kw)
get_params( param_pars={},  **kw)
load( load_pars=None,  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
reset_model(  )
save( model=None, session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_sklearn\model_sklearn.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset( data_pars=None,  **kw)
get_params( param_pars={},  **kw)
load( load_pars={},  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
reset_model(  )
save( model=None, session=None, save_pars={},  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_sklearn\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\nbeats.py
----------------methods----------------

---------------functions---------------
data_generator(x_full, y_full, bs,   )
eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test,   )
load(model, optimiser,   )
plot_scatter(  *args, **kwargs)
save(model, optimiser, grad_step,   )
train_100_grad_steps(data, device, net, optimiser,   )


mlmodels\model_sklearn\raw\atspy\atspy\pyaf.py
----------------methods----------------

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\ssa.py
----------------methods----------------
mySSA.__init__(self, time_series,   )
mySSA._dot(x, y,   )
mySSA._forecast_prep(self,  singular_values=None,  )
mySSA.decompose(self,  verbose=False,  )
mySSA.diagonal_averaging(hankel_matrix,   )
mySSA.embed(self,  embedding_dimension=None, suspected_frequency=None, verbose=False, return_df=False,  )
mySSA.forecast_recurrent(self,  steps_ahead=12, singular_values=None, plot=False, return_df=False,  **plotargs)
mySSA.get_contributions( X=None, s=None, plot=True,  )
mySSA.split(arr, size,   )
mySSA.view_reconstruction(cls,  names=None, return_df=False, plot=True, symmetric_plots=False,  *hankel)
mySSA.view_s_contributions(self,  adjust_scale=False, cumulative=False, return_df=False,  )
mySSA.view_time_series(self,   )

---------------functions---------------
data_generator(x_full, y_full, bs,   )
eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test,   )
load(model, optimiser,   )
plot_scatter(  *args, **kwargs)
save(model, optimiser, grad_step,   )
train_100_grad_steps(data, device, net, optimiser,   )


mlmodels\model_sklearn\raw\atspy\atspy\__init__.py
----------------methods----------------

---------------functions---------------
add_freq(idx,  freq=None,  )
constant_feature_detect(data,  threshold=0.98,  )
ensemble_doubled(middle_in, middle_out, forecast_in, forecast_out,   )
ensemble_first(middle_in, forecast_in,   )
ensemble_lightgbm(forecast_in, forecast_out, pred,   )
ensemble_performance(forecasts,   )
ensemble_pure(forecast_in, forecast_out,   )
ensemble_tsfresh(forecast_in, forecast_out, season, perd,   )
forecast_frame(test, forecast_dict,   )
forecast_frame_insample(forecast_dict, test,   )
forecast_frame_outsample(forecast_dict, df, forecast_len, index,   )
forecast_models(models_dict, forecast_len, freq, df,  in_sample=True, GPU=False,  )
get_unique_N(iterable, N,   )
gluonts_dataframe(df,   )
infer_periodocity(train,   )
infer_seasonality(train,  index=0,  )
infer_seasonality_ssa(train,  index=1,  )
insample_performance(test, forecast_dict,  dict=False,  )
middle(ensemble_lgb, ensemble_ts, pure_ensemble,   )
nbeats_dataframe(df, forecast_length, in_sample, device,  train_portion=0.75,  )
original_dataframe(df, freq,   )
parse_data(df,   )
prophet_dataframe(df,   )
season_list(train,   )
select_seasonality(train, season,   )
time_feature(df, perd,   )
train_models(train, models, forecast_len,  full_df=None, seasonality="infer_from_data", in_sample=None, freq=None, GPU=None,  )
train_test_split(df,  train_proportion=0.75,  )


mlmodels\model_sklearn\raw\atspy\atspy\TS\Exogenous.py
----------------methods----------------
cExogenousInfo.__init__(self,   )
cExogenousInfo.addVars(self, df,   )
cExogenousInfo.createEncodedExogenous(self,   )
cExogenousInfo.fit(self,   )
cExogenousInfo.info(self,   )
cExogenousInfo.to_json(self,   )
cExogenousInfo.transformDataset(self, df,   )
cExogenousInfo.updateExogenousVariableInfo(self,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\Keras_Models.py
----------------methods----------------
cAbstract_RNN_Model.__init__(self, cycle_residue_name, P,  iExogenousInfo=None,  )
cAbstract_RNN_Model.build_RNN_Architecture(self,   )
cAbstract_RNN_Model.dumpCoefficients(self,  iMax=10,  )
cAbstract_RNN_Model.fit(self,   )
cAbstract_RNN_Model.reshape_inputs(self, iInputs,   )
cAbstract_RNN_Model.transformDataset(self, df,  horizon_index=1,  )
cLSTM_Model.__getstate__(self,   )
cLSTM_Model.__init__(self, cycle_residue_name, P,  iExogenousInfo=None,  )
cLSTM_Model.__setstate__(self, istate,   )
cLSTM_Model.build_RNN_Architecture(self,   )
cLSTM_Model.build_RNN_Architecture_template(self,   )
cLSTM_Model.reshape_target(self, iTarget,   )
cLSTM_Model.testPickle(self,   )
cLSTM_Model.testPickle_old(self,   )
cMLP_Model.__getstate__(self,   )
cMLP_Model.__init__(self, cycle_residue_name, P,  iExogenousInfo=None,  )
cMLP_Model.__setstate__(self, istate,   )
cMLP_Model.build_RNN_Architecture(self,   )
cMLP_Model.build_RNN_Architecture_template(self,   )
cMLP_Model.reshape_target(self, iTarget,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\MissingData.py
----------------methods----------------
cMissingDataImputer.__init__(self,   )
cMissingDataImputer.has_missing_data(self, iSeries,   )
cMissingDataImputer.interpolate_signal_if_needed(self, iInputDS, iSignal,   )
cMissingDataImputer.interpolate_time_if_needed(self, iInputDS, iTime,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\Options.py
----------------methods----------------
cCrossValidationOptions.__init__(self,   )
cCrostonOptions.__init__(self,   )
cModelControl.__init__(self,   )
cModelControl.disable_all_autoregressions(self,   )
cModelControl.disable_all_periodics(self,   )
cModelControl.disable_all_transformations(self,   )
cModelControl.disable_all_trends(self,   )
cModelControl.set_active_autoregressions(self, autoregs,   )
cModelControl.set_active_periodics(self, periodics,   )
cModelControl.set_active_transformations(self, transformations,   )
cModelControl.set_active_trends(self, trends,   )
cSignalDecomposition_Options.__init__(self,   )
cSignalDecomposition_Options.canBuildKerasModel(self, iModel,   )
cSignalDecomposition_Options.disableDebuggingOptions(self,   )
cSignalDecomposition_Options.enable_fast_mode(self,   )
cSignalDecomposition_Options.enable_low_memory_mode(self,   )
cSignalDecomposition_Options.enable_slow_mode(self,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\Perf.py
----------------methods----------------
cPerf.__init__(self,   )
cPerf.check_not_nan(self, sig, name,   )
cPerf.compute(self, signal, estimator, name,   )
cPerf.computeCriterion(self, signal, estimator, criterion, name,   )
cPerf.compute_MAPE_SMAPE_MASE(self, signal, estimator,   )
cPerf.compute_R2(self, signal, estimator,   )
cPerf.compute_pearson_r(self, signal, estimator,   )
cPerf.dump_perf_data(self, signal, estimator,   )
cPerf.getCriterionValue(self, criterion,   )
cPerf.real_compute(self, signal, estimator, name,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\Plots.py
----------------methods----------------

---------------functions---------------
add_patched_legend(ax, names,   )
build_record_label(labels_list,   )
decomp_plot(df, time, signal, estimator, residue,  name=None, format='png', max_length=1000,  )
decomp_plot_as_png_base64(df, time, signal, estimator, residue,  name=None, max_length=1000,  )
plot_hierarchy(structure, iAnnotations, name,   )
prediction_interval_plot(df, time, signal, estimator, lower, upper,  name=None, format='png', max_length=1000,  )
prediction_interval_plot_as_png_base64(df, time, signal, estimator, lower, upper,  name=None, max_length=1000,  )
qqplot_residues(df, residue,   )


mlmodels\model_sklearn\raw\atspy\atspy\TS\PredictionIntervals.py
----------------methods----------------
cPredictionIntervalsEstimator.__init__(self,   )
cPredictionIntervalsEstimator.computePerformances(self,   )
cPredictionIntervalsEstimator.dump(self,   )
cPredictionIntervalsEstimator.dump_detailed(self,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\Scikit_Models.py
----------------methods----------------
cAbstract_Scikit_Model.__init__(self, cycle_residue_name, P,  iExogenousInfo=None,  )
cAbstract_Scikit_Model.build_Scikit_Model(self,   )
cAbstract_Scikit_Model.dumpCoefficients(self,  iMax=10,  )
cAbstract_Scikit_Model.fit(self,   )
cAbstract_Scikit_Model.set_name(self,   )
cAbstract_Scikit_Model.transformDataset(self, df,  horizon_index=1,  )
cAutoRegressiveModel.__init__(self, cycle_residue_name, P,  iExogenousInfo=None,  )
cAutoRegressiveModel.build_Scikit_Model(self,   )
cAutoRegressiveModel.dumpCoefficients(self,  iMax=10,  )
cAutoRegressiveModel.set_name(self,   )
cSVR_Model.__init__(self, cycle_residue_name, P,  iExogenousInfo=None,  )
cSVR_Model.build_Scikit_Model(self,   )
cSVR_Model.dumpCoefficients(self,  iMax=10,  )
cSVR_Model.set_name(self,   )
cXGBoost_Model.__init__(self, cycle_residue_name, P,  iExogenousInfo=None,  )
cXGBoost_Model.build_Scikit_Model(self,   )
cXGBoost_Model.dumpCoefficients(self,  iMax=10,  )
cXGBoost_Model.get_default_xgb_options(self,   )
cXGBoost_Model.set_name(self,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\SignalDecomposition.py
----------------methods----------------
cPerf_Arg.__init__(self, name,   )
cSignalDecomposition.__init__(self,   )
cSignalDecomposition.checkData(self, iInputDS, iTime, iSignal, iHorizon, iExogenousData,   )
cSignalDecomposition.forecast(self, iInputDS, iHorizon,   )
cSignalDecomposition.getModelFormula(self,   )
cSignalDecomposition.getModelInfo(self,   )
cSignalDecomposition.getPlotsAsDict(self,   )
cSignalDecomposition.standardPlots(self,  name=None, format='png',  )
cSignalDecomposition.to_json(self,   )
cSignalDecomposition.train(self, iInputDS, iTime, iSignal, iHorizon,  iExogenousData=None,  )
cSignalDecompositionOneTransform.__init__(self,   )
cSignalDecompositionOneTransform.collectPerformanceIndices(self,   )
cSignalDecompositionOneTransform.computePerfsInParallel(self, args,   )
cSignalDecompositionOneTransform.run_gc(self,   )
cSignalDecompositionOneTransform.serialize(self,   )
cSignalDecompositionOneTransform.setParams(self, iInputDS, iTime, iSignal, iHorizon, iTransformation,  iExogenousData=None,  )
cSignalDecompositionOneTransform.train(self, iInputDS, iTime, iSignal, iHorizon, iTransformation,   )
cSignalDecompositionOneTransform.updatePerfsForAllModels(self, iModels,   )
cSignalDecompositionTrainer.__init__(self,   )
cSignalDecompositionTrainer.cleanup_after_model_selection(self,   )
cSignalDecompositionTrainer.collectPerformanceIndices(self,   )
cSignalDecompositionTrainer.collectPerformanceIndices_ModelSelection(self,   )
cSignalDecompositionTrainer.defineTransformations(self, df, iTime, iSignal,   )
cSignalDecompositionTrainer.perform_model_selection(self,   )
cSignalDecompositionTrainer.train(self, iInputDS, iTime, iSignal, iHorizon,   )
cSignalDecompositionTrainer.train_multiprocessed(self, iInputDS, iTime, iSignal, iHorizon,   )
cSignalDecompositionTrainer.train_not_threaded(self, iInputDS, iTime, iSignal, iHorizon,   )
cSignalDecompositionTrainer.train_threaded(self, iInputDS, iTime, iSignal, iHorizon,   )
cSignalDecompositionTrainer.validateTransformation(self, transf, df, iTime, iSignal,   )
cSignalDecompositionTrainer_CrossValidation.__init__(self,   )
cSignalDecompositionTrainer_CrossValidation.define_splits(self,   )
cSignalDecompositionTrainer_CrossValidation.perform_model_selection(self,   )
cSignalDecompositionTrainer_CrossValidation.train(self, iInputDS, iTime, iSignal, iHorizon,   )
cTraining_Arg.__init__(self, name,   )

---------------functions---------------
compute_perf_func(arg,   )
run_transform_thread(arg,   )


mlmodels\model_sklearn\raw\atspy\atspy\TS\SignalDecomposition_AR.py
----------------methods----------------
cAbstractAR.__init__(self, cycle_residue_name,  iExogenousInfo=None,  )
cAbstractAR.addLagForForecast(self, df, lag_df, series, p,   )
cAbstractAR.computePerf(self,   )
cAbstractAR.dumpCoefficients(self,   )
cAbstractAR.generateLagsForForecast(self, df,   )
cAbstractAR.getDefaultValue(self, series,   )
cAbstractAR.plot(self,   )
cAbstractAR.shift_series(self, series, p, idefault,   )
cAutoRegressiveEstimator.__init__(self,   )
cAutoRegressiveEstimator.addLagForTraining(self, df, lag_df, series, autoreg, p,   )
cAutoRegressiveEstimator.addLagsForTraining(self, df, cycle_residue,  iNeedExogenous=False,  )
cAutoRegressiveEstimator.check_not_nan(self, sig, name,   )
cAutoRegressiveEstimator.estimate(self,   )
cAutoRegressiveEstimator.estimate_ar_models_for_cycle(self, cycle_residue,   )
cAutoRegressiveEstimator.is_not_constant(self, iSeries,   )
cAutoRegressiveEstimator.plotAR(self,   )
cAutoRegressiveEstimator.shift_series(self, series, p,   )
cZeroAR.__init__(self, cycle_residue_name,   )
cZeroAR.fit(self,   )
cZeroAR.transformDataset(self, df,  horizon_index=1,  )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\SignalDecomposition_Cycle.py
----------------methods----------------
cAbstractCycle.__init__(self, trend,   )
cAbstractCycle.check_not_nan(self, sig, name,   )
cAbstractCycle.computePerf(self,   )
cAbstractCycle.getCycleResidueName(self,   )
cAbstractCycle.plot(self,   )
cBestCycleForTrend.__init__(self, trend, criterion,   )
cBestCycleForTrend.computeBestCycle(self,   )
cBestCycleForTrend.dumpCyclePerfs(self,   )
cBestCycleForTrend.fit(self,   )
cBestCycleForTrend.generate_cycles(self,   )
cBestCycleForTrend.getCycleName(self,   )
cBestCycleForTrend.transformDataset(self, df,   )
cCycleEstimator.__init__(self,   )
cCycleEstimator.addSeasonal(self, trend, seas_type, resolution,   )
cCycleEstimator.defineCycles(self,   )
cCycleEstimator.dumpCyclePerf(self, cycle,   )
cCycleEstimator.estimateAllCycles(self,   )
cCycleEstimator.estimateCycles(self,   )
cCycleEstimator.filterSeasonals(self,   )
cCycleEstimator.plotCycles(self,   )
cSeasonalPeriodic.__init__(self, trend, date_part,   )
cSeasonalPeriodic.fit(self,   )
cSeasonalPeriodic.getCycleName(self,   )
cSeasonalPeriodic.get_date_part(self, x,   )
cSeasonalPeriodic.get_date_part_encoding(self, x,   )
cSeasonalPeriodic.hasEnoughData(self, iTimeMin, iTimeMax,   )
cSeasonalPeriodic.transformDataset(self, df,   )
cZeroCycle.__init__(self, trend,   )
cZeroCycle.fit(self,   )
cZeroCycle.getCycleName(self,   )
cZeroCycle.transformDataset(self, df,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\SignalDecomposition_Quant.py
----------------methods----------------
cSignalQuantizer.__init__(self,   )
cSignalQuantizer.quant2signal(self, series, iSignal, Q,   )
cSignalQuantizer.quantizeSignal(self, iSignal, Q,   )
cSignalQuantizer.signal2quant(self, x, curve,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\SignalDecomposition_Trend.py
----------------methods----------------
cAbstractTrend.__init__(self,   )
cAbstractTrend.check_not_nan(self, sig, name,   )
cAbstractTrend.computePerf(self,   )
cConstantTrend.__init__(self,   )
cConstantTrend.addTrendInputVariables(self,   )
cConstantTrend.compute(self,   )
cConstantTrend.fit(self,   )
cConstantTrend.transformDataset(self, df,   )
cLag1Trend.__init__(self,   )
cLag1Trend.addTrendInputVariables(self,   )
cLag1Trend.compute(self,   )
cLag1Trend.fit(self,   )
cLag1Trend.replaceFirstMissingValue(self, df, series,   )
cLag1Trend.transformDataset(self, df,   )
cLinearTrend.__init__(self,   )
cLinearTrend.addTrendInputVariables(self,   )
cLinearTrend.compute(self,   )
cLinearTrend.fit(self,   )
cLinearTrend.transformDataset(self, df,   )
cMovingAverageTrend.__init__(self, iWindow,   )
cMovingAverageTrend.addTrendInputVariables(self,   )
cMovingAverageTrend.compute(self,   )
cMovingAverageTrend.fit(self,   )
cMovingAverageTrend.transformDataset(self, df,   )
cMovingMedianTrend.__init__(self, iWindow,   )
cMovingMedianTrend.addTrendInputVariables(self,   )
cMovingMedianTrend.compute(self,   )
cMovingMedianTrend.fit(self,   )
cMovingMedianTrend.transformDataset(self, df,   )
cPolyTrend.__init__(self,   )
cPolyTrend.addTrendInputVariables(self,   )
cPolyTrend.compute(self,   )
cPolyTrend.fit(self,   )
cPolyTrend.transformDataset(self, df,   )
cTrendEstimator.__init__(self,   )
cTrendEstimator.addTrendInputVariables(self,   )
cTrendEstimator.check_residue(self, sig, name,   )
cTrendEstimator.defineTrends(self,   )
cTrendEstimator.estimateTrend(self,   )
cTrendEstimator.estimateTrends(self,   )
cTrendEstimator.needMovingTrend(self, df, i,   )
cTrendEstimator.plotTrend(self,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\SignalHierarchy.py
----------------methods----------------
cSignalHierarchy.__init__(self,   )
cSignalHierarchy.addVars(self, df,   )
cSignalHierarchy.checkData(self, df,   )
cSignalHierarchy.computeBottomUpForecast(self, iForecast_DF, level, signal,  iPrefix="BU",  )
cSignalHierarchy.computeBottomUpForecasts(self, iForecast_DF,   )
cSignalHierarchy.computeMiddleOutForecasts(self, iForecast_DF, iProp, iPrefix,   )
cSignalHierarchy.computeOptimalCombination(self, iForecast_DF,   )
cSignalHierarchy.computePerfOnCombinedForecasts(self, iForecast_DF,   )
cSignalHierarchy.computeTopDownForecastedProportions(self, iForecast_DF,   )
cSignalHierarchy.computeTopDownForecasts(self, iForecast_DF, iProp, iPrefix,   )
cSignalHierarchy.computeTopDownHistoricalProportions(self, iAllLevelsDataset,   )
cSignalHierarchy.create_HierarchicalStructure(self,   )
cSignalHierarchy.create_SummingMatrix(self,   )
cSignalHierarchy.create_all_levels_dataset(self, df,   )
cSignalHierarchy.create_all_levels_models(self, iAllLevelsDataset, H, iDateColumn,   )
cSignalHierarchy.fit(self,   )
cSignalHierarchy.forecast(self, iInputDS, iHorizon,   )
cSignalHierarchy.forecastAllModels(self, iAllLevelsDataset, H, iDateColumn,   )
cSignalHierarchy.getEstimPart(self, df,   )
cSignalHierarchy.getModelInfo(self,   )
cSignalHierarchy.getValidPart(self, df,   )
cSignalHierarchy.info(self,   )
cSignalHierarchy.internal_forecast(self, iInputDS, iHorizon,   )
cSignalHierarchy.plot(self,  name=None,  )
cSignalHierarchy.standardPlots(self,  name=None,  )
cSignalHierarchy.to_json(self,   )
cSignalHierarchy.transformDataset(self, df,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\Signal_Grouping.py
----------------methods----------------
cSignalGrouping.__init__(self,   )
cSignalGrouping.add_level(self, previous_level,   )
cSignalGrouping.create_HierarchicalStructure(self,   )
cSignalGrouping.tuple_to_string(self, k,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\Signal_Transformation.py
----------------methods----------------
cAbstractSignalTransform.__init__(self,   )
cAbstractSignalTransform.apply(self, sig,   )
cAbstractSignalTransform.check_not_nan(self, sig, name,   )
cAbstractSignalTransform.dump_apply_invert(self, df_before_apply, df_after_apply,   )
cAbstractSignalTransform.fit(self, sig,   )
cAbstractSignalTransform.fit_scaling_params(self, sig,   )
cAbstractSignalTransform.invert(self, sig1,   )
cAbstractSignalTransform.is_applicable(self, sig,   )
cAbstractSignalTransform.rescale_signal(self, sig1,   )
cAbstractSignalTransform.rescale_value(self, x,   )
cAbstractSignalTransform.scale_signal(self, sig,   )
cAbstractSignalTransform.scale_value(self, x,   )
cAbstractSignalTransform.test(self,   )
cAbstractSignalTransform.transformDataset(self, df, isig,   )
cSignalTransform_Accumulate.__init__(self,   )
cSignalTransform_Accumulate.get_name(self, iSig,   )
cSignalTransform_Accumulate.specific_apply(self, sig,   )
cSignalTransform_Accumulate.specific_fit(self, sig,   )
cSignalTransform_Accumulate.specific_invert(self, df,   )
cSignalTransform_Anscombe.__init__(self,   )
cSignalTransform_Anscombe.get_name(self, iSig,   )
cSignalTransform_Anscombe.specific_apply(self, sig,   )
cSignalTransform_Anscombe.specific_fit(self, sig,   )
cSignalTransform_Anscombe.specific_invert(self, sig,   )
cSignalTransform_BoxCox.__init__(self, iLambda,   )
cSignalTransform_BoxCox.get_name(self, iSig,   )
cSignalTransform_BoxCox.invert_value(self, y,   )
cSignalTransform_BoxCox.specific_apply(self, df,   )
cSignalTransform_BoxCox.specific_fit(self, sig,   )
cSignalTransform_BoxCox.specific_invert(self, df,   )
cSignalTransform_Differencing.__init__(self,   )
cSignalTransform_Differencing.get_name(self, iSig,   )
cSignalTransform_Differencing.specific_apply(self, df,   )
cSignalTransform_Differencing.specific_fit(self, sig,   )
cSignalTransform_Differencing.specific_invert(self, df,   )
cSignalTransform_Fisher.__init__(self,   )
cSignalTransform_Fisher.get_name(self, iSig,   )
cSignalTransform_Fisher.specific_apply(self, sig,   )
cSignalTransform_Fisher.specific_fit(self, sig,   )
cSignalTransform_Fisher.specific_invert(self, sig,   )
cSignalTransform_Logit.__init__(self,   )
cSignalTransform_Logit.get_name(self, iSig,   )
cSignalTransform_Logit.inv_logit(self, y,   )
cSignalTransform_Logit.is_applicable(self, sig,   )
cSignalTransform_Logit.logit(self, x,   )
cSignalTransform_Logit.specific_apply(self, df,   )
cSignalTransform_Logit.specific_fit(self, sig,   )
cSignalTransform_Logit.specific_invert(self, df,   )
cSignalTransform_None.__init__(self,   )
cSignalTransform_None.get_name(self, iSig,   )
cSignalTransform_None.specific_apply(self, df,   )
cSignalTransform_None.specific_fit(self, sig,   )
cSignalTransform_None.specific_invert(self, df,   )
cSignalTransform_Quantize.__init__(self, iQuantiles,   )
cSignalTransform_Quantize.get_name(self, iSig,   )
cSignalTransform_Quantize.is_applicable(self, sig,   )
cSignalTransform_Quantize.quant2signal(self, x,   )
cSignalTransform_Quantize.signal2quant(self, x,   )
cSignalTransform_Quantize.specific_apply(self, df,   )
cSignalTransform_Quantize.specific_fit(self, sig,   )
cSignalTransform_Quantize.specific_invert(self, df,   )
cSignalTransform_RelativeDifferencing.__init__(self,   )
cSignalTransform_RelativeDifferencing.get_name(self, iSig,   )
cSignalTransform_RelativeDifferencing.specific_apply(self, df,   )
cSignalTransform_RelativeDifferencing.specific_fit(self, sig,   )
cSignalTransform_RelativeDifferencing.specific_invert(self, df,   )

---------------functions---------------
create_tranformation(iName, arg,   )
testTransform(tr1,   )
testTransform_one_seed(tr1, seed_value,   )


mlmodels\model_sklearn\raw\atspy\atspy\TS\Time.py
----------------methods----------------

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\TimeSeriesModel.py
----------------methods----------------
cTimeSeriesModel.__init__(self, transf, trend, cycle, autoreg,   )
cTimeSeriesModel.addPredictionIntervals(self, iInputDS, iForecastFrame, iHorizon,   )
cTimeSeriesModel.applyForecastRectifier(self, df,   )
cTimeSeriesModel.computePredictionIntervals(self,   )
cTimeSeriesModel.forecast(self, df, iHorizon,   )
cTimeSeriesModel.forecastOneStepAhead(self, df,  horizon_index=1, perf_mode=False,  )
cTimeSeriesModel.getComplexity(self,   )
cTimeSeriesModel.getForecastDatasetForPlots(self,   )
cTimeSeriesModel.getFormula(self,   )
cTimeSeriesModel.getInfo(self,   )
cTimeSeriesModel.getPlotsAsDict(self,   )
cTimeSeriesModel.getVersions(self,   )
cTimeSeriesModel.get_model_category(self,   )
cTimeSeriesModel.plotForecasts(self, df,   )
cTimeSeriesModel.plotResidues(self,  name=None, format='png',  )
cTimeSeriesModel.signal_info(self,   )
cTimeSeriesModel.standardPlots(self,  name=None, format='png',  )
cTimeSeriesModel.to_json(self,   )
cTimeSeriesModel.updatePerfs(self,  compute_all_indicators=False,  )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\TimeSeries_Cutting.py
----------------methods----------------
cCuttingInfo.__init__(self,   )
cCuttingInfo.check_split(self, iSplit,   )
cCuttingInfo.cutFrame(self, df,   )
cCuttingInfo.defineCuttingParameters(self,   )
cCuttingInfo.estimate(self,   )
cCuttingInfo.getEstimPart(self, df,   )
cCuttingInfo.getValidPart(self, df,   )
cCuttingInfo.info(self,   )
cCuttingInfo.set_default_split(self,   )
cCuttingInfo.set_split(self, iSplit,   )

---------------functions---------------


mlmodels\model_sklearn\raw\atspy\atspy\TS\Utils.py
----------------methods----------------
Internal_PyAF_Error.__init__(self, reason,   )
PyAF_Error.__init__(self, reason,   )
cMemoize.__call__(self,   *args)
cMemoize.__get__(self, obj, objtype,   )
cMemoize.__init__(self, f,   )

---------------functions---------------
createDirIfNeeded(dirname,   )
get_pyaf_hierarchical_logger(  )
get_pyaf_logger(  )


mlmodels\model_sklearn\raw\atspy\atspy\TS\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\03_nbeats_dataloader.py
----------------methods----------------

---------------functions---------------
Model(model_pars, data_pars, compute_pars,   )
data_generator(x_full, y_full, bs,   )
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model, data_pars, compute_pars, out_pars,   )
fit_simple(net, optimiser, data_generator, on_save_callback, device, data_pars, out_pars,  max_grad_steps=500,  )
get_dataset(data_pars,   )
get_params(param_pars,   **kw)
load(load_pars,   )
load_checkpoint(model, optimiser,  CHECKPOINT_NAME="nbeats-fiting-checkpoint.th",  )
plot(net, x, target, backcast_length, forecast_length, grad_step,  out_path="./",  )
plot_model(net, x, target, grad_step, data_pars,  disable_plot=False,  )
plot_predict(x_test, y_test, p, data_pars, compute_pars, out_pars,   )
predict(model, sess,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
save(model, session, save_pars,   )
save_checkpoint(model, optimiser, grad_step,  CHECKPOINT_NAME="mycheckpoint",  )
test( data_path="dataset/milk.csv",  )


mlmodels\model_tch\matchzoo_models.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None, out_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  )
get_config_file(  )
get_dataset( data_pars=None,  **kw)
get_dataset_wikiqa(data_pars, model,   )
get_params( param_pars=None,  **kw)
load(load_pars,   )
predict(model,  session=None, data_pars=None, compute_pars=None, out_pars=None,  )
save(model,  session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_tch\mlp.py
----------------methods----------------
Model.__init__(self,   )
Model.forward(self, x,   )

---------------functions---------------


mlmodels\model_tch\nbeats.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
Model.split(arr, size,   )

---------------functions---------------
data_generator(x_full, y_full, bs,   )
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_simple(net, optimiser, data_generator, on_save_callback, device, data_pars, out_pars,  max_grad_steps=500,  )
get_data(data_pars,   )
get_dataset(  **kw)
get_params(param_pars,   **kw)
load(load_pars,   )
load_checkpoint(model, optimiser,  CHECKPOINT_NAME='nbeats-fiting-checkpoint.th',  )
plot(net, x, target, backcast_length, forecast_length, grad_step,  out_path="./",  )
plot_model(net, x, target, grad_step, data_pars,  disable_plot=False,  )
plot_predict(x_test, y_test, p, data_pars, compute_pars, out_pars,   )
predict(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
save(model, session, save_pars,   )
save_checkpoint(model, optimiser, grad_step,  CHECKPOINT_NAME="mycheckpoint",  )
test( choice="json", data_path="nbeats.json", config_mode="test",  )


mlmodels\model_tch\pplm.py
----------------methods----------------

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
generate(cond_text, bag_of_words,  discrim=None, class_label=-1,  )
get_dataset( data_pars=None,  **kw)
get_params( param_pars=None,  **kw)
path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)


mlmodels\model_tch\pytorch_vae.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None, out_pars=None,  )

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  )
get_dataset( data_pars=None,  **kw)
get_params( param_pars=None,  **kw)
load(load_pars,   )
predict(model,  session=None, data_pars=None, compute_pars=None, out_pars=None, imax=1, return_ytrue=1,  )
save(model,  session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_tch\textcnn.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )
TextCNN.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
TextCNN.forward(self, x,   )
TextCNN.rebuild_embed(self, vocab_built,   )
TextCNN.tokenizer(text,   )

---------------functions---------------
_get_device(  )
_train(m, device, train_itr, optimizer, epoch, max_epoch,   )
_valid(m, device, test_itr,   )
clean_str(string,   )
create_data_iterator(tr_batch_size, val_batch_size, tabular_train, tabular_valid, d,   )
create_tabular_dataset(path_train, path_valid,  lang='en', pretrained_emb='glove.6B.300d',  )
fit(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
fit_metrics(model,  session=None, data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
get_config_file(  )
get_data_file(  )
get_dataset( data_pars=None, out_pars=None,  **kwargs)
get_params( param_pars=None,  **kw)
load( load_pars=None,  )
predict(model,  session=None, data_pars=None, compute_pars=None, out_pars=None, return_ytrue=1,  )
save(model,  session=None, save_pars=None,  )
split_train_valid(path_data, path_train, path_valid,  frac=0.7,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_tch\torchhub.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None, out_pars=None,  )

---------------functions---------------
_get_device(  )
_train(m, device, train_itr, criterion, optimizer, epoch, max_epoch,  imax=1,  )
_valid(m, device, test_itr, criterion,  imax=1,  )
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwargs)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  )
get_config_file(  )
get_dataset( data_pars=None,  **kw)
get_params( param_pars=None,  **kw)
load(load_pars,   )
predict(model,  session=None, data_pars=None, compute_pars=None, out_pars=None, imax=1, return_ytrue=1,  )
save(model,  session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )
test2( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\model_tch\transformer_classifier.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  )

---------------functions---------------
_preprocess_XXXX(df,   **kw)
fit(train_dataset, model, tokenizer,   )
fit_metrics(model, tokenizer, model_pars, data_pars, out_pars, compute_pars,  prefix="",  )
get_dataset(task, tokenizer,  evaluate=False,  )
get_eval_report(labels, preds,   )
get_mismatched(labels, preds,   )
get_params( param_pars={},  **kw)
load( load_pars={},  )
load_and_cache_examples(task, tokenizer,  evaluate=False,  )
metrics(task_name, preds, labels,   )
reset_model(  )
save( model=None, session=None, save_pars={},  )
test(data_path, model_pars, data_pars, compute_pars, out_pars,  pars_choice=0,  )


mlmodels\model_tch\transformer_sentence.py
----------------methods----------------

---------------functions---------------
fit(model,  data_pars=None, model_pars=None, compute_pars=None, out_pars=None,  *args, **kw)
fit_metrics(model,  session=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset( data_pars=None,  **kw)
get_params(param_pars,   **kw)
load( load_pars=None,  )
predict(model,  session=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
reset_model(  )
save(model,  session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="test01",  )


mlmodels\model_tch\util_data.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\util_transformer.py
----------------methods----------------
BinaryProcessor._create_examples(self, lines, set_type,   )
BinaryProcessor.get_dev_examples(self, data_dir,   )
BinaryProcessor.get_labels(self,   )
BinaryProcessor.get_train_examples(self, data_dir,   )
DataProcessor._read_tsv(cls, input_file,  quotechar=None,  )
DataProcessor.get_dev_examples(self, data_dir,   )
DataProcessor.get_labels(self,   )
DataProcessor.get_train_examples(self, data_dir,   )
InputExample.__init__(self, guid, text_a,  text_b=None, label=None,  )
InputFeatures.__init__(self, input_ids, input_mask, segment_ids, label_id,   )

---------------functions---------------
_truncate_seq_pair(tokens_a, tokens_b, max_length,   )
convert_example_to_feature(example_row,  pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True, sep_token_extra=False,  )
convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode,  cls_token_at_end=False, sep_token_extra=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0, sequence_b_segment_id=1, cls_token_segment_id=1, pad_token_segment_id=0, mask_padding_with_zero=True, process_count=cpu_count(),  )


mlmodels\model_tch\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\model_chatbot\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\model_chatbot\diag_gpt\Chatbot_run.py
----------------methods----------------

---------------functions---------------
generate(  )
get_bot_response(  )
home(  )
recalc( p=None,  )
reinput(user_msg,   )
top_p_filtering(logits,  top_p=0.9, filter_value=-float('Inf'),  )


mlmodels\model_tch\model_chatbot\diag_gpt\myChatbot.py
----------------methods----------------

---------------functions---------------
generate(  )
get_bot_response(  )
home(  )
recalc( p=None,  )
reinput(user_msg,   )
top_p_filtering(logits,  top_p=0.9, filter_value=-float('Inf'),  )


mlmodels\model_tch\raw\01_cnn_classifier.py
----------------methods----------------
Model.__init__(self,  model_pars=NOne, data_pars=None, compute_pars=None,  )
Model.forward(self, x,   )
Model.to(  *arg, **kwarg)

---------------functions---------------
_eval_metrics(model, epoch, test_loader,   )
_log_scalar(name, value, step,   )
_train(epoch, model, train_loader,   )
cli_load_arguments(  )
fit(model, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
get_dataset(data_params,   )
get_params( choice="test",  **kwargs)
metrics(model,  sess=None, data_params={}, compute_params={},  )
predict(model,  sess=None, compute_params=None, data_params=None,  )
predict_proba(model,  sess=None, compute_params=None, data_params=None,  )
test(arg,   )


mlmodels\model_tch\raw\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\nbeats\data.py
----------------methods----------------

---------------functions---------------
get_data(num_samples, backcast_length, forecast_length,  signal_type='seasonality', random=False,  )


mlmodels\model_tch\raw\nbeats\model.py
----------------methods----------------
Block.__init__(self, units, thetas_dim, device,  backcast_length=10, forecast_length=5, share_thetas=False,  )
Block.__str__(self,   )
Block.forward(self, x,   )
GenericBlock.__init__(self, units, thetas_dim, device,  backcast_length=10, forecast_length=5,  )
GenericBlock.forward(self, x,   )
NBeatsNet.__init__(self, device,  stack_types=[TREND_BLOCK, SEASONALITY_BLOCK], nb_blocks_per_stack=3, forecast_length=5, backcast_length=10, thetas_dims=[4, 8], share_weights_in_stack=False, hidden_layer_units=256,  )
NBeatsNet.create_stack(self, stack_id,   )
NBeatsNet.forward(self, backcast,   )
NBeatsNet.select_block(block_type,   )
SeasonalityBlock.__init__(self, units, thetas_dim, device,  backcast_length=10, forecast_length=5,  )
SeasonalityBlock.forward(self, x,   )
TrendBlock.__init__(self, units, thetas_dim, device,  backcast_length=10, forecast_length=5,  )
TrendBlock.forward(self, x,   )

---------------functions---------------
linspace(backcast_length, forecast_length,   )
seasonality_model(thetas, t, device,   )
trend_model(thetas, t, device,   )


mlmodels\model_tch\raw\nbeats\nbeats.py
----------------methods----------------
Model.__init__(self,  learning_rate=0.001, num_layers=2, size=None, size_layer=128, output_size=None, forget_bias=0.1, timestep=5, epoch=5,  )
Model.plot_model(x, target, grad_step,   )

---------------functions---------------
fit(model, data_pars,   )
fit(model, data_pars,   )
get_script_arguments(  )
load(model, optimiser,   )
plot(net, x, target, backcast_length, forecast_length, grad_step,   )
predict(model, sess, data_pars,   )
reset_model(  )
save(model, optimiser, grad_step,   )
simple_fit(net, optimiser, data_generator, on_save_callback, device,  max_grad_steps=10000,  )
stats_compute(model, sess, df,   )


mlmodels\model_tch\raw\nbeats\nbeats_sample.py
----------------methods----------------

---------------functions---------------
data_generator(x_full, y_full, bs,   )
data_load(  )
eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test,   )
fit(  )
plot_scatter(  *args, **kwargs)
train_100_grad_steps(data, device, net, optimiser, test_losses,   )


mlmodels\model_tch\raw\nbeats\trainer.py
----------------methods----------------

---------------functions---------------
fit(  )
get_script_arguments(  )
load(model, optimiser,   )
plot(net, x, target, backcast_length, forecast_length, grad_step,   )
save(model, optimiser, grad_step,   )
simple_fit(net, optimiser, data_generator, on_save_callback, device,  max_grad_steps=10000,  )


mlmodels\model_tch\raw\nbeats\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pplm\pplm_classification_head.py
----------------methods----------------
ClassificationHead.__init__(self, class_size, embed_size,   )
ClassificationHead.forward(self, hidden_state,   )

---------------functions---------------


mlmodels\model_tch\raw\pplm\run_pplm.py
----------------methods----------------

---------------functions---------------
full_text_generation(model, tokenizer,  context=None, num_samples=1, device="cuda", bag_of_words=None, discrim=None, class_label=None, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=True, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, verbosity_level=REGULAR,  **kwargs)
generate_text_pplm(model, tokenizer,  context=None, past=None, device="cuda", perturb=True, bow_indices=None, classifier=None, class_label=None, loss_type=0, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=True, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, verbosity_level=REGULAR,  )
get_bag_of_words_indices(bag_of_words_ids_or_paths,   )
get_classifier(name,   )
perturb_past(past, model, last,  unpert_past=None, unpert_logits=None, accumulated_hidden=None, grad_norms=None, stepsize=0.01, one_hot_bows_vectors=None, classifier=None, class_label=None, loss_type=0, num_iterations=3, horizon_length=1, window_length=0, decay=False, gamma=1.5, kl_scale=0.01, device='cuda', verbosity_level=REGULAR,  )
run_pplm_example( pretrained_model="gpt2-medium", cond_text="", uncond=False, num_samples=1, bag_of_words=None, discrim=None, discrim_weights=None, discrim_meta=None, class_label=-1, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=True, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, seed=0, no_cuda=False, colorama=False, verbosity='regular',  )
set_generic_model_params(discrim_weights, discrim_meta,   )
to_var(x,  requires_grad=False, volatile=False, device='cuda',  )
top_k_filter(logits, k,  probs=False,  )


mlmodels\model_tch\raw\pplm\run_pplm_discrim_train.py
----------------methods----------------
Dataset.__getitem__(self, index,   )
Dataset.__init__(self, X, y,   )
Dataset.__len__(self,   )
Dataset.pad_sequences(sequences,   )
Discriminator.__init__(self, class_size,  pretrained_model="gpt2-medium", cached_mode=False, device='cpu',  )
Discriminator.avg_representation(self, x,   )
Discriminator.forward(self, x,   )
Discriminator.get_classifier(self,   )
Discriminator.train_custom(self,   )

---------------functions---------------
cached_collate_fn(data,   )
collate_fn(data,   )
evaluate_performance(data_loader, discriminator,  device='cpu',  )
get_cached_data_loader(dataset, batch_size, discriminator,  shuffle=False, device='cpu',  )
predict(input_sentence, model, classes,  cached=False, device='cpu',  )
train_discriminator(dataset,  dataset_fp=None, pretrained_model="gpt2-medium", epochs=10, batch_size=64, log_interval=10, save_model=False, cached=False, no_cuda=False,  )
train_epoch(data_loader, discriminator, optimizer,  epoch=0, log_interval=10, device='cpu',  )


mlmodels\model_tch\raw\pplm\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pplm\paper_code\gpt2tunediscrim.py
----------------methods----------------
ClassificationHead.__init__(self,  class_size=5, embed_size=2048,  )
ClassificationHead.forward(self, hidden_state,   )
Dataset.__getitem__(self, index,   )
Dataset.__init__(self, X, y,   )
Dataset.__len__(self,   )
Dataset.merge(sequences,   )
Discriminator.__init__(self,   )
Discriminator.forward(self, x,   )
Discriminator.train(self,   )
Discriminator2.__init__(self,  class_size=5, embed_size=1024,  )
Discriminator2.forward(self, x,   )
Discriminator2.get_classifier(self,   )
Discriminator2.train_custom(self,   )
Discriminator2mean.__init__(self,  class_size=5, embed_size=1024,  )
Discriminator2mean.forward(self, x,   )
Discriminator2mean.get_classifier(self,   )
Discriminator2mean.train_custom(self,   )

---------------functions---------------
collate_fn(data,   )
main(  )
test_epoch(data_loader, discriminator,  device='cuda', args=None,  )
train_epoch(data_loader, discriminator,  device='cuda', args=None, epoch=1,  )


mlmodels\model_tch\raw\pplm\paper_code\pplm.py
----------------methods----------------

---------------functions---------------
latent_perturb(model, args,  context=None, sample=True, device='cuda',  )
perturb_past(past, model, prev, args, classifier,  good_index=None, stepsize=0.01, vocab_size=50257, original_probs=None, accumulated_hidden=None, true_past=None, grad_norms=None,  )
run_model(  )
sample_from_hidden(model, args, classifier,  context=None, past=None, device='cuda', sample=True, perturb=True, good_index=None,  )


mlmodels\model_tch\raw\pplm\paper_code\style_utils.py
----------------methods----------------

---------------functions---------------
to_var(x,  requires_grad=False, volatile=False,  )
top_k_logits(logits, k,  probs=False,  )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\convert_gpt2_checkpoint_to_pytorch.py
----------------methods----------------

---------------functions---------------
convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\convert_openai_checkpoint_to_pytorch.py
----------------methods----------------

---------------functions---------------
convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\convert_tf_checkpoint_to_pytorch.py
----------------methods----------------

---------------functions---------------
convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\convert_transfo_xl_checkpoint_to_pytorch.py
----------------methods----------------

---------------functions---------------
convert_transfo_xl_checkpoint_to_pytorch(tf_checkpoint_path, transfo_xl_config_file, pytorch_dump_folder_path, transfo_xl_dataset_file,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\file_utils.py
----------------methods----------------

---------------functions---------------
cached_path(url_or_filename,  cache_dir=None,  )
filename_to_url(filename,  cache_dir=None,  )
get_file_extension(path,  dot=True, lower=True,  )
get_from_cache(url,  cache_dir=None,  )
http_get(url, temp_file,   )
read_set_from_file(filename,   )
s3_etag(url,   )
s3_get(url, temp_file,   )
s3_request(func,   )
split_s3_path(url,   )
url_to_filename(url,  etag=None,  )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\modeling.py
----------------methods----------------
BertAttention.__init__(self, config,   )
BertAttention.forward(self, input_tensor, attention_mask,   )
BertConfig.__init__(self, vocab_size_or_config_json_file,  hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02,  )
BertConfig.__repr__(self,   )
BertConfig.from_dict(cls, json_object,   )
BertConfig.from_json_file(cls, json_file,   )
BertConfig.to_dict(self,   )
BertConfig.to_json_file(self, json_file_path,   )
BertConfig.to_json_string(self,   )
BertEmbeddings.__init__(self, config,   )
BertEmbeddings.forward(self, input_ids,  token_type_ids=None,  )
BertEncoder.__init__(self, config,   )
BertEncoder.forward(self, hidden_states, attention_mask,  output_all_encoded_layers=True,  )
BertForMaskedLM.__init__(self, config,   )
BertForMaskedLM.forward(self, input_ids,  token_type_ids=None, attention_mask=None, masked_lm_labels=None,  )
BertForMultipleChoice.__init__(self, config, num_choices,   )
BertForMultipleChoice.forward(self, input_ids,  token_type_ids=None, attention_mask=None, labels=None,  )
BertForNextSentencePrediction.__init__(self, config,   )
BertForNextSentencePrediction.forward(self, input_ids,  token_type_ids=None, attention_mask=None, next_sentence_label=None,  )
BertForPreTraining.__init__(self, config,   )
BertForPreTraining.forward(self, input_ids,  token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None,  )
BertForQuestionAnswering.__init__(self, config,   )
BertForQuestionAnswering.forward(self, input_ids,  token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,  )
BertForSequenceClassification.__init__(self, config, num_labels,   )
BertForSequenceClassification.forward(self, input_ids,  token_type_ids=None, attention_mask=None, labels=None,  )
BertForTokenClassification.__init__(self, config, num_labels,   )
BertForTokenClassification.forward(self, input_ids,  token_type_ids=None, attention_mask=None, labels=None,  )
BertIntermediate.__init__(self, config,   )
BertIntermediate.forward(self, hidden_states,   )
BertLMPredictionHead.__init__(self, config, bert_model_embedding_weights,   )
BertLMPredictionHead.forward(self, hidden_states,   )
BertLayer.__init__(self, config,   )
BertLayer.forward(self, hidden_states, attention_mask,   )
BertLayerNorm.__init__(self, hidden_size,  eps=1e-12,  )
BertLayerNorm.forward(self, x,   )
BertModel.__init__(self, config,   )
BertModel.forward(self, input_ids,  token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,  )
BertOnlyMLMHead.__init__(self, config, bert_model_embedding_weights,   )
BertOnlyMLMHead.forward(self, sequence_output,   )
BertOnlyNSPHead.__init__(self, config,   )
BertOnlyNSPHead.forward(self, pooled_output,   )
BertOutput.__init__(self, config,   )
BertOutput.forward(self, hidden_states, input_tensor,   )
BertPooler.__init__(self, config,   )
BertPooler.forward(self, hidden_states,   )
BertPreTrainedModel.__init__(self, config,   *inputs, **kwargs)
BertPreTrainedModel.from_pretrained(cls, pretrained_model_name_or_path,  state_dict=None, cache_dir=None, from_tf=False,  *inputs, **kwargs)
BertPreTrainedModel.init_bert_weights(self, module,   )
BertPreTrainingHeads.__init__(self, config, bert_model_embedding_weights,   )
BertPreTrainingHeads.forward(self, sequence_output, pooled_output,   )
BertPredictionHeadTransform.__init__(self, config,   )
BertPredictionHeadTransform.forward(self, hidden_states,   )
BertSelfAttention.__init__(self, config,   )
BertSelfAttention.forward(self, hidden_states, attention_mask,   )
BertSelfAttention.transpose_for_scores(self, x,   )
BertSelfOutput.__init__(self, config,   )
BertSelfOutput.forward(self, hidden_states, input_tensor,   )

---------------functions---------------
gelu(x,   )
load_tf_weights_in_bert(model, tf_checkpoint_path,   )
swish(x,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\modeling_gpt2.py
----------------methods----------------
Attention.__init__(self, nx, n_ctx, config,  scale=False,  )
Attention._attn(self, q, k, v,   )
Attention.forward(self, x,  layer_past=None,  )
Attention.merge_heads(self, x,   )
Attention.split_heads(self, x,  k=False,  )
Block.__init__(self, n_ctx, config,  scale=False,  )
Block.forward(self, x,  layer_past=None,  )
Conv1D.__init__(self, nf, nx,   )
Conv1D.forward(self, x,   )
GPT2Config.__init__(self,  vocab_size_or_config_json_file=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, layer_norm_epsilon=1e-5, initializer_range=0.02,  )
GPT2Config.__repr__(self,   )
GPT2Config.from_dict(cls, json_object,   )
GPT2Config.from_json_file(cls, json_file,   )
GPT2Config.to_dict(self,   )
GPT2Config.to_json_file(self, json_file_path,   )
GPT2Config.to_json_string(self,   )
GPT2DoubleHeadsModel.__init__(self, config,   )
GPT2DoubleHeadsModel.forward(self, input_ids, mc_token_ids,  lm_labels=None, mc_labels=None, token_type_ids=None, position_ids=None, past=None,  )
GPT2DoubleHeadsModel.set_tied(self,   )
GPT2LMHead.__init__(self, model_embeddings_weights, config,   )
GPT2LMHead.forward(self, hidden_state,   )
GPT2LMHead.set_embeddings_weights(self, model_embeddings_weights,   )
GPT2LMHeadModel.__init__(self, config,   )
GPT2LMHeadModel.forward(self, input_ids,  position_ids=None, token_type_ids=None, lm_labels=None, past=None,  )
GPT2LMHeadModel.forward_embed(self, inputs_ids,  position_ids=None, token_type_ids=None, past=None,  )
GPT2LMHeadModel.forward_hidden(self, hidden_states,   )
GPT2LMHeadModel.forward_transformer_embed(self, hidden_states,  past=None, add_one=False,  )
GPT2LMHeadModel.set_tied(self,   )
GPT2Model.__init__(self, config,   )
GPT2Model.forward(self, input_ids,  position_ids=None, token_type_ids=None, past=None,  )
GPT2Model.forward_embed(self, input_ids,  position_ids=None, token_type_ids=None, past=None,  )
GPT2Model.forward_transformer(self, hidden_states,  past=None, add_one=False,  )
GPT2MultipleChoiceHead.__init__(self, config,   )
GPT2MultipleChoiceHead.forward(self, hidden_states, mc_token_ids,   )
GPT2PreTrainedModel.__init__(self, config,   *inputs, **kwargs)
GPT2PreTrainedModel.from_pretrained(cls, pretrained_model_name_or_path,  state_dict=None, cache_dir=None, from_tf=False,  *inputs, **kwargs)
GPT2PreTrainedModel.init_weights(self, module,   )
GPT2PreTrainedModel.set_tied(self,   )
MLP.__init__(self, n_state, config,   )
MLP.forward(self, x,   )

---------------functions---------------
gelu(x,   )
load_tf_weights_in_gpt2(model, gpt2_checkpoint_path,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\modeling_openai.py
----------------methods----------------
Attention.__init__(self, nx, n_ctx, config,  scale=False,  )
Attention._attn(self, q, k, v,   )
Attention.forward(self, x,   )
Attention.merge_heads(self, x,   )
Attention.split_heads(self, x,  k=False,  )
Block.__init__(self, n_ctx, config,  scale=False,  )
Block.forward(self, x,   )
Conv1D.__init__(self, nf, rf, nx,   )
Conv1D.forward(self, x,   )
MLP.__init__(self, n_state, config,   )
MLP.forward(self, x,   )
OpenAIGPTConfig.__init__(self,  vocab_size_or_config_json_file=40478, n_special=0, n_positions=512, n_ctx=512, n_embd=768, n_layer=12, n_head=12, afn="gelu", resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1, layer_norm_epsilon=1e-5, initializer_range=0.02,  )
OpenAIGPTConfig.__repr__(self,   )
OpenAIGPTConfig.from_dict(cls, json_object,   )
OpenAIGPTConfig.from_json_file(cls, json_file,   )
OpenAIGPTConfig.to_dict(self,   )
OpenAIGPTConfig.to_json_file(self, json_file_path,   )
OpenAIGPTConfig.to_json_string(self,   )
OpenAIGPTConfig.total_tokens_embeddings(self,   )
OpenAIGPTDoubleHeadsModel.__init__(self, config,   )
OpenAIGPTDoubleHeadsModel.forward(self, input_ids, mc_token_ids,  lm_labels=None, mc_labels=None, token_type_ids=None, position_ids=None,  )
OpenAIGPTDoubleHeadsModel.set_num_special_tokens(self, num_special_tokens,   )
OpenAIGPTLMHead.__init__(self, model_embeddings_weights, config,   )
OpenAIGPTLMHead.forward(self, hidden_state,   )
OpenAIGPTLMHead.set_embeddings_weights(self, model_embeddings_weights,   )
OpenAIGPTLMHeadModel.__init__(self, config,   )
OpenAIGPTLMHeadModel.forward(self, input_ids,  position_ids=None, token_type_ids=None, lm_labels=None,  )
OpenAIGPTLMHeadModel.set_num_special_tokens(self, num_special_tokens,   )
OpenAIGPTModel.__init__(self, config,   )
OpenAIGPTModel.forward(self, input_ids,  position_ids=None, token_type_ids=None,  )
OpenAIGPTModel.set_num_special_tokens(self, num_special_tokens,   )
OpenAIGPTMultipleChoiceHead.__init__(self, config,   )
OpenAIGPTMultipleChoiceHead.forward(self, hidden_states, mc_token_ids,   )
OpenAIGPTPreTrainedModel.__init__(self, config,   *inputs, **kwargs)
OpenAIGPTPreTrainedModel.from_pretrained(cls, pretrained_model_name_or_path,  num_special_tokens=None, state_dict=None, cache_dir=None, from_tf=False,  *inputs, **kwargs)
OpenAIGPTPreTrainedModel.init_weights(self, module,   )
OpenAIGPTPreTrainedModel.set_num_special_tokens(self, num_special_tokens,   )

---------------functions---------------
gelu(x,   )
load_tf_weights_in_openai_gpt(model, openai_checkpoint_folder_path,   )
swish(x,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\modeling_transfo_xl.py
----------------methods----------------
AdaptiveEmbedding.__init__(self, n_token, d_embed, d_proj, cutoffs,  div_val=1, sample_softmax=False,  )
AdaptiveEmbedding.forward(self, inp,   )
DecoderLayer.__init__(self, n_head, d_model, d_head, d_inner, dropout,   **kwargs)
DecoderLayer.forward(self, dec_inp,  dec_attn_mask=None, mems=None,  )
MultiHeadAttn.__init__(self, n_head, d_model, d_head, dropout,  dropatt=0, pre_lnorm=False, r_r_bias=None, r_w_bias=None,  )
MultiHeadAttn.forward(self, h,  attn_mask=None, mems=None,  )
PositionalEmbedding.__init__(self, demb,   )
PositionalEmbedding.forward(self, pos_seq,  bsz=None,  )
PositionwiseFF.__init__(self, d_model, d_inner, dropout,  pre_lnorm=False,  )
PositionwiseFF.forward(self, inp,   )
RelLearnableDecoderLayer.__init__(self, n_head, d_model, d_head, d_inner, dropout,   **kwargs)
RelLearnableDecoderLayer.forward(self, dec_inp, r_emb, r_w_bias, r_bias,  dec_attn_mask=None, mems=None,  )
RelLearnableMultiHeadAttn.__init__(self,   *args, **kwargs)
RelLearnableMultiHeadAttn.forward(self, w, r_emb, r_w_bias, r_bias,  attn_mask=None, mems=None,  )
RelMultiHeadAttn.__init__(self, n_head, d_model, d_head, dropout,  dropatt=0, tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False, r_r_bias=None, r_w_bias=None,  )
RelMultiHeadAttn._parallelogram_mask(self, h, w,  left=False,  )
RelMultiHeadAttn._rel_shift(self, x,  zero_triu=False,  )
RelMultiHeadAttn._shift(self, x, qlen, klen, mask,  left=False,  )
RelMultiHeadAttn.forward(self, w, r,  attn_mask=None, mems=None,  )
RelPartialLearnableDecoderLayer.__init__(self, n_head, d_model, d_head, d_inner, dropout,   **kwargs)
RelPartialLearnableDecoderLayer.forward(self, dec_inp, r,  dec_attn_mask=None, mems=None,  )
RelPartialLearnableMultiHeadAttn.__init__(self,   *args, **kwargs)
RelPartialLearnableMultiHeadAttn.forward(self, w, r,  attn_mask=None, mems=None,  )
TransfoXLConfig.__init__(self,  vocab_size_or_config_json_file=267735, cutoffs=[20000, 40000, 200000], d_model=1024, d_embed=1024, n_head=16, d_head=64, d_inner=4096, div_val=4, pre_lnorm=False, n_layer=18, tgt_len=128, ext_len=0, mem_len=1600, clamp_len=1000, same_length=True, proj_share_all_but_first=True, attn_type=0, sample_softmax=-1, adaptive=True, tie_weight=True, dropout=0.1, dropatt=0.0, untie_r=True, init="normal", init_range=0.01, proj_init_std=0.01, init_std=0.02,  )
TransfoXLConfig.__repr__(self,   )
TransfoXLConfig.from_dict(cls, json_object,   )
TransfoXLConfig.from_json_file(cls, json_file,   )
TransfoXLConfig.to_dict(self,   )
TransfoXLConfig.to_json_file(self, json_file_path,   )
TransfoXLConfig.to_json_string(self,   )
TransfoXLLMHeadModel.__init__(self, config,   )
TransfoXLLMHeadModel.forward(self, input_ids, target,  mems=None,  )
TransfoXLLMHeadModel.init_mems(self, data,   )
TransfoXLLMHeadModel.reset_length(self, tgt_len, ext_len, mem_len,   )
TransfoXLLMHeadModel.tie_weights(self,   )
TransfoXLModel.__init__(self, config,   )
TransfoXLModel._forward(self, dec_inp,  mems=None,  )
TransfoXLModel._update_mems(self, hids, mems, qlen, mlen,   )
TransfoXLModel.backward_compatible(self,   )
TransfoXLModel.forward(self, input_ids,  mems=None,  )
TransfoXLModel.init_mems(self, data,   )
TransfoXLModel.reset_length(self, tgt_len, ext_len, mem_len,   )
TransfoXLPreTrainedModel.__init__(self, config,   *inputs, **kwargs)
TransfoXLPreTrainedModel.from_pretrained(cls, pretrained_model_name_or_path,  state_dict=None, cache_dir=None, from_tf=False,  *inputs, **kwargs)
TransfoXLPreTrainedModel.init_bias(self, bias,   )
TransfoXLPreTrainedModel.init_weight(self, weight,   )
TransfoXLPreTrainedModel.init_weights(self, m,   )
TransfoXLPreTrainedModel.set_num_special_tokens(self, num_special_tokens,   )

---------------functions---------------
build_tf_to_pytorch_map(model, config,   )
load_tf_weights_in_transfo_xl(model, config, tf_path,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\modeling_transfo_xl_utilities.py
----------------methods----------------
LogUniformSampler.__init__(self, range_max, n_sample,   )
LogUniformSampler.sample(self, labels,   )
ProjectedAdaptiveLogSoftmax.__init__(self, n_token, d_embed, d_proj, cutoffs,  div_val=1, keep_order=False,  )
ProjectedAdaptiveLogSoftmax._compute_logit(self, hidden, weight, bias, proj,   )
ProjectedAdaptiveLogSoftmax.forward(self, hidden,  target=None, keep_order=False,  )
ProjectedAdaptiveLogSoftmax.log_prob(self, hidden,   )

---------------functions---------------
sample_logits(embedding, bias, labels, inputs, sampler,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\optimization.py
----------------methods----------------
BertAdam.__init__(self, params,  lr=required, warmup=-1, t_total=-1, schedule='warmup_linear', b1=0.9, b2=0.999, e=1e-6, weight_decay=0.01, max_grad_norm=1.0,  )
BertAdam.get_lr(self,   )
BertAdam.step(self,  closure=None,  )

---------------functions---------------
warmup_constant(x,  warmup=0.002,  )
warmup_cosine(x,  warmup=0.002,  )
warmup_linear(x,  warmup=0.002,  )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\optimization_openai.py
----------------methods----------------
OpenAIAdam.__init__(self, params,  lr=required, schedule='warmup_linear', warmup=-1, t_total=-1, b1=0.9, b2=0.999, e=1e-8, weight_decay=0, vector_l2=False, max_grad_norm=-1,  **kwargs)
OpenAIAdam.get_lr(self,   )
OpenAIAdam.step(self,  closure=None,  )

---------------functions---------------
warmup_constant(x,  warmup=0.002,  )
warmup_cosine(x,  warmup=0.002,  )
warmup_linear(x,  warmup=0.002,  )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\tokenization.py
----------------methods----------------
BasicTokenizer.__init__(self,  do_lower_case=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"),  )
BasicTokenizer._clean_text(self, text,   )
BasicTokenizer._is_chinese_char(self, cp,   )
BasicTokenizer._run_split_on_punc(self, text,   )
BasicTokenizer._run_strip_accents(self, text,   )
BasicTokenizer._tokenize_chinese_chars(self, text,   )
BasicTokenizer.tokenize(self, text,   )
BertTokenizer.__init__(self, vocab_file,  do_lower_case=True, max_len=None, do_basic_tokenize=True, never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"),  )
BertTokenizer.convert_ids_to_tokens(self, ids,   )
BertTokenizer.convert_tokens_to_ids(self, tokens,   )
BertTokenizer.from_pretrained(cls, pretrained_model_name_or_path,  cache_dir=None,  *inputs, **kwargs)
BertTokenizer.save_vocabulary(self, vocab_path,   )
BertTokenizer.tokenize(self, text,   )
WordpieceTokenizer.__init__(self, vocab,  unk_token="[UNK]", max_input_chars_per_word=100,  )
WordpieceTokenizer.tokenize(self, text,   )

---------------functions---------------
_is_control(char,   )
_is_punctuation(char,   )
_is_whitespace(char,   )
load_vocab(vocab_file,   )
whitespace_tokenize(text,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\tokenization_gpt2.py
----------------methods----------------
GPT2Tokenizer.__init__(self, vocab_file, merges_file,  errors='replace', special_tokens=None, max_len=None,  )
GPT2Tokenizer.__len__(self,   )
GPT2Tokenizer.bpe(self, token,   )
GPT2Tokenizer.convert_ids_to_tokens(self, ids,  skip_special_tokens=False,  )
GPT2Tokenizer.convert_tokens_to_ids(self, tokens,   )
GPT2Tokenizer.decode(self, tokens,   )
GPT2Tokenizer.encode(self, text,   )
GPT2Tokenizer.from_pretrained(cls, pretrained_model_name_or_path,  cache_dir=None,  *inputs, **kwargs)
GPT2Tokenizer.lru_cache(  )
GPT2Tokenizer.save_vocabulary(self, vocab_path,   )
GPT2Tokenizer.set_special_tokens(self, special_tokens,   )
GPT2Tokenizer.tokenize(self, text,   )

---------------functions---------------
bytes_to_unicode(  )
get_pairs(word,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\tokenization_openai.py
----------------methods----------------
OpenAIGPTTokenizer.__init__(self, vocab_file, merges_file,  special_tokens=None, max_len=None,  )
OpenAIGPTTokenizer.__len__(self,   )
OpenAIGPTTokenizer.bpe(self, token,   )
OpenAIGPTTokenizer.convert_ids_to_tokens(self, ids,  skip_special_tokens=False,  )
OpenAIGPTTokenizer.convert_tokens_to_ids(self, tokens,   )
OpenAIGPTTokenizer.decode(self, ids,  skip_special_tokens=False, clean_up_tokenization_spaces=True,  )
OpenAIGPTTokenizer.encode(self, text,   )
OpenAIGPTTokenizer.from_pretrained(cls, pretrained_model_name_or_path,  cache_dir=None,  *inputs, **kwargs)
OpenAIGPTTokenizer.save_vocabulary(self, vocab_path,   )
OpenAIGPTTokenizer.set_special_tokens(self, special_tokens,   )
OpenAIGPTTokenizer.tokenize(self, text,   )

---------------functions---------------
get_pairs(word,   )
text_standardize(text,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\tokenization_transfo_xl.py
----------------methods----------------
LMMultiFileIterator.__init__(self, paths, vocab, bsz, bptt,  device='cpu', ext_len=None, shuffle=False,  )
LMMultiFileIterator.__iter__(self,   )
LMMultiFileIterator.get_sent_stream(self, path,   )
LMOrderedIterator.__init__(self, data, bsz, bptt,  device='cpu', ext_len=None,  )
LMOrderedIterator.__iter__(self,   )
LMOrderedIterator.get_batch(self, i,  bptt=None,  )
LMOrderedIterator.get_fixlen_iter(self,  start=0,  )
LMOrderedIterator.get_varlen_iter(self,  start=0, std=5, min_len=5, max_deviation=3,  )
LMShuffledIterator.__init__(self, data, bsz, bptt,  device='cpu', ext_len=None, shuffle=False,  )
LMShuffledIterator.__iter__(self,   )
LMShuffledIterator.get_sent_stream(self,   )
LMShuffledIterator.stream_iterator(self, sent_stream,   )
TransfoXLCorpus.__init__(self,   *args, **kwargs)
TransfoXLCorpus.build_corpus(self, path, dataset,   )
TransfoXLCorpus.from_pretrained(cls, pretrained_model_name_or_path,  cache_dir=None,  *inputs, **kwargs)
TransfoXLCorpus.get_iterator(self, split,   *args, **kwargs)
TransfoXLTokenizer.__init__(self,  special=[], min_freq=0, max_size=None, lower_case=False, delimiter=None, vocab_file=None, never_split=("<unk>", "<eos>", "<formula>"),  )
TransfoXLTokenizer.__len__(self,   )
TransfoXLTokenizer._build_from_file(self, vocab_file,   )
TransfoXLTokenizer.add_special(self, sym,   )
TransfoXLTokenizer.add_symbol(self, sym,   )
TransfoXLTokenizer.build_vocab(self,   )
TransfoXLTokenizer.convert_ids_to_tokens(self, indices,   )
TransfoXLTokenizer.convert_to_tensor(self, symbols,   )
TransfoXLTokenizer.convert_tokens_to_ids(self, symbols,   )
TransfoXLTokenizer.count_file(self, path,  verbose=False, add_eos=False,  )
TransfoXLTokenizer.count_sents(self, sents,  verbose=False,  )
TransfoXLTokenizer.decode(self, indices,  exclude=None,  )
TransfoXLTokenizer.encode_file(self, path,  ordered=False, verbose=False, add_eos=True, add_double_eos=False,  )
TransfoXLTokenizer.encode_sents(self, sents,  ordered=False, verbose=False,  )
TransfoXLTokenizer.from_pretrained(cls, pretrained_model_name_or_path,  cache_dir=None,  *inputs, **kwargs)
TransfoXLTokenizer.get_idx(self, sym,   )
TransfoXLTokenizer.get_sym(self, idx,   )
TransfoXLTokenizer.save_vocabulary(self, vocab_path,   )
TransfoXLTokenizer.tokenize(self, line,  add_eos=False, add_double_eos=False,  )

---------------functions---------------
get_lm_corpus(datadir, dataset,   )


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pplm\paper_code\pytorch_pretrained_bert\__main__.py
----------------methods----------------

---------------functions---------------
main(  )


mlmodels\model_tch\raw\pplm\pplm_transformer\pplm.py
----------------methods----------------
ClassificationHead.__init__(self, class_size, embed_size,   )
ClassificationHead.forward(self, hidden_state,   )
Model.__init__(self, args, device,   )
Model.add(  *k, **kw)

---------------functions---------------
full_text_generation(model, tokenizer,  context=None, num_samples=1, device="cuda", bag_of_words=None, discrim=None, class_label=None, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=False, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, repetition_penalty=1.0,  **kwargs)
generate_text_pplm(model, tokenizer,  context=None, past=None, device="cuda", perturb=True, bow_indices=None, classifier=None, class_label=None, loss_type=0, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=False, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, repetition_penalty=1.0,  )
get_bag_of_words_indices(bag_of_words_ids_or_paths,   )
get_classifier(name,   )
perturb_past(past, model, last,  unpert_past=None, unpert_logits=None, accumulated_hidden=None, grad_norms=None, stepsize=0.01, one_hot_bows_vectors=None, classifier=None, class_label=None, loss_type=0, num_iterations=3, horizon_length=1, window_length=0, decay=False, gamma=1.5, kl_scale=0.01, device="cuda",  )
run_pplm_example( pretrained_model="gpt2-medium", cond_text="", uncond=False, num_samples=1, bag_of_words=None, discrim=None, discrim_weights=None, discrim_meta=None, class_label=-1, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=False, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, seed=0, no_cuda=False, colorama=False, repetition_penalty=1.0,  )
set_generic_model_params(discrim_weights, discrim_meta,   )
to_var(x,  requires_grad=False, volatile=False, device="cuda",  )
top_k_filter(logits, k,  probs=False,  )


mlmodels\model_tch\raw\pplm\pplm_transformer\pplm_classification_head.py
----------------methods----------------
ClassificationHead.__init__(self, class_size, embed_size,   )
ClassificationHead.forward(self, hidden_state,   )

---------------functions---------------


mlmodels\model_tch\raw\pplm\pplm_transformer\pplm_train.py
----------------methods----------------
Dataset.__getitem__(self, index,   )
Dataset.__init__(self, X, y,   )
Dataset.__len__(self,   )
Dataset.pad_sequences(sequences,   )
Discriminator.__init__(self, class_size,  pretrained_model="gpt2-medium", cached_mode=False, device="cpu",  )
Discriminator.avg_representation(self, x,   )
Discriminator.forward(self, x,   )
Discriminator.get_classifier(self,   )
Discriminator.train_custom(self,   )

---------------functions---------------
cached_collate_fn(data,   )
collate_fn(data,   )
evaluate_performance(data_loader, discriminator,  device="cpu",  )
get_cached_data_loader(dataset, batch_size, discriminator,  shuffle=False, device="cpu",  )
predict(input_sentence, model, classes,  cached=False, device="cpu",  )
train_discriminator(dataset,  dataset_fp=None, pretrained_model="gpt2-medium", epochs=10, batch_size=64, log_interval=10, save_model=False, cached=False, no_cuda=False,  )
train_epoch(data_loader, discriminator, optimizer,  epoch=0, log_interval=10, device="cpu",  )


mlmodels\model_tch\raw\pplm\pplm_transformer\run_pplm.py
----------------methods----------------

---------------functions---------------
full_text_generation(model, tokenizer,  context=None, num_samples=1, device="cuda", bag_of_words=None, discrim=None, class_label=None, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=False, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, repetition_penalty=1.0,  **kwargs)
generate_text_pplm(model, tokenizer,  context=None, past=None, device="cuda", perturb=True, bow_indices=None, classifier=None, class_label=None, loss_type=0, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=False, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, repetition_penalty=1.0,  )
get_bag_of_words_indices(bag_of_words_ids_or_paths,   )
get_classifier(name,   )
perturb_past(past, model, last,  unpert_past=None, unpert_logits=None, accumulated_hidden=None, grad_norms=None, stepsize=0.01, one_hot_bows_vectors=None, classifier=None, class_label=None, loss_type=0, num_iterations=3, horizon_length=1, window_length=0, decay=False, gamma=1.5, kl_scale=0.01, device="cuda",  )
run_pplm_example( pretrained_model="gpt2-medium", cond_text="", uncond=False, num_samples=1, bag_of_words=None, discrim=None, discrim_weights=None, discrim_meta=None, class_label=-1, length=100, stepsize=0.02, temperature=1.0, top_k=10, sample=False, num_iterations=3, grad_length=10000, horizon_length=1, window_length=0, decay=False, gamma=1.5, gm_scale=0.9, kl_scale=0.01, seed=0, no_cuda=False, colorama=False, repetition_penalty=1.0,  )
set_generic_model_params(discrim_weights, discrim_meta,   )
to_var(x,  requires_grad=False, volatile=False, device="cuda",  )
top_k_filter(logits, k,  probs=False,  )


mlmodels\model_tch\raw\pplm\pplm_transformer\run_pplm_discrim_train.py
----------------methods----------------
Dataset.__getitem__(self, index,   )
Dataset.__init__(self, X, y,   )
Dataset.__len__(self,   )
Dataset.pad_sequences(sequences,   )
Discriminator.__init__(self, class_size,  pretrained_model="gpt2-medium", cached_mode=False, device="cpu",  )
Discriminator.avg_representation(self, x,   )
Discriminator.forward(self, x,   )
Discriminator.get_classifier(self,   )
Discriminator.train_custom(self,   )

---------------functions---------------
cached_collate_fn(data,   )
collate_fn(data,   )
evaluate_performance(data_loader, discriminator,  device="cpu",  )
get_cached_data_loader(dataset, batch_size, discriminator,  shuffle=False, device="cpu",  )
predict(input_sentence, model, classes,  cached=False, device="cpu",  )
train_discriminator(dataset,  dataset_fp=None, pretrained_model="gpt2-medium", epochs=10, batch_size=64, log_interval=10, save_model=False, cached=False, no_cuda=False,  )
train_epoch(data_loader, discriminator, optimizer,  epoch=0, log_interval=10, device="cpu",  )


mlmodels\model_tch\raw\pplm\pplm_transformer\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\aae\aae.py
----------------methods----------------
Decoder.__init__(self,   )
Decoder.forward(self, z,   )
Discriminator.__init__(self,   )
Discriminator.forward(self, z,   )
Encoder.__init__(self,   )
Encoder.forward(self, img,   )

---------------functions---------------
reparameterization(mu, logvar,   )
sample_image(n_row, batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\acgan\acgan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, noise, labels,   )

---------------functions---------------
sample_image(n_row, batches_done,   )
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\began\began.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, noise,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\bgan\bgan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------
boundary_seeking_loss(y_pred, y_true,   )


mlmodels\model_tch\raw\pytorch_gan\models\bicyclegan\bicyclegan.py
----------------methods----------------

---------------functions---------------
reparameterization(mu, logvar,   )
sample_images(batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\bicyclegan\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root, input_shape,  mode="train",  )
ImageDataset.__len__(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\bicyclegan\models.py
----------------methods----------------
Encoder.__init__(self, latent_dim, input_shape,   )
Encoder.forward(self, img,   )
Generator.__init__(self, latent_dim, img_shape,   )
Generator.forward(self, x, z,   )
MultiDiscriminator.__init__(self, input_shape,   )
MultiDiscriminator.compute_loss(self, x, gt,   )
MultiDiscriminator.forward(self, x,   )
UNetDown.__init__(self, in_size, out_size,  normalize=True, dropout=0.0,  )
UNetDown.forward(self, x,   )
UNetUp.__init__(self, in_size, out_size,   )
UNetUp.forward(self, x, skip_input,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\ccgan\ccgan.py
----------------methods----------------

---------------functions---------------
apply_random_mask(imgs,   )
save_sample(saved_samples,   )


mlmodels\model_tch\raw\pytorch_gan\models\ccgan\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root,  transforms_x=None, transforms_lr=None, mode='train',  )
ImageDataset.__len__(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\ccgan\models.py
----------------methods----------------
Discriminator.__init__(self, input_shape,   )
Discriminator.forward(self, img,   )
Generator.__init__(self, input_shape,   )
Generator.forward(self, x, x_lr,   )
UNetDown.__init__(self, in_size, out_size,  normalize=True, dropout=0.0,  )
UNetDown.forward(self, x,   )
UNetUp.__init__(self, in_size, out_size,  dropout=0.0,  )
UNetUp.forward(self, x, skip_input,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\cgan\cgan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img, labels,   )
Generator.__init__(self,   )
Generator.forward(self, noise, labels,   )

---------------functions---------------
sample_image(n_row, batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\cluster_gan\clustergan.py
----------------methods----------------
Discriminator_CNN.__init__(self, latent_dim, n_c, x_shape,  verbose=False,  )
Discriminator_CNN.forward(self, zn, zc,   )
Encoder_CNN.__init__(self,  shape=[],  )
Encoder_CNN.extra_repr(self,   )
Encoder_CNN.forward(self, x,   )

---------------functions---------------
calc_gradient_penalty(netD, real_data, generated_data,   )
initialize_weights(net,   )
sample_z( shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False,  )
softmax(x,   )


mlmodels\model_tch\raw\pytorch_gan\models\cogan\cogan.py
----------------methods----------------
CoupledDiscriminators.__init__(self,   )
CoupledDiscriminators.forward(self, img1, img2,   )
CoupledGenerators.__init__(self,   )
CoupledGenerators.forward(self, noise,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\cogan\mnistm.py
----------------methods----------------
MNISTM.__getitem__(self, index,   )
MNISTM.__init__(self, root,  mnist_root="data", train=True, transform=None, target_transform=None, download=False,  )
MNISTM.__len__(self,   )
MNISTM._check_exists(self,   )
MNISTM.download(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\context_encoder\context_encoder.py
----------------methods----------------

---------------functions---------------
save_sample(batches_done,   )
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\context_encoder\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root,  transforms_=None, img_size=128, mask_size=64, mode="train",  )
ImageDataset.__len__(self,   )
ImageDataset.apply_center_mask(self, img,   )
ImageDataset.apply_random_mask(self, img,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\context_encoder\models.py
----------------methods----------------
Discriminator.__init__(self,  channels=3,  )
Discriminator.forward(self, img,   )
Generator.__init__(self,  channels=3,  )
Generator.forward(self, x,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\cyclegan\cyclegan.py
----------------methods----------------

---------------functions---------------
sample_images(batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\cyclegan\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root,  transforms_=None, unaligned=False, mode="train",  )
ImageDataset.__len__(self,   )

---------------functions---------------
to_rgb(image,   )


mlmodels\model_tch\raw\pytorch_gan\models\cyclegan\models.py
----------------methods----------------
Discriminator.__init__(self, input_shape,   )
Discriminator.forward(self, img,   )
GeneratorResNet.__init__(self, input_shape, num_residual_blocks,   )
GeneratorResNet.forward(self, x,   )
ResidualBlock.__init__(self, in_features,   )
ResidualBlock.forward(self, x,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\cyclegan\utils.py
----------------methods----------------
LambdaLR.__init__(self, n_epochs, offset, decay_start_epoch,   )
LambdaLR.step(self, epoch,   )
ReplayBuffer.__init__(self,  max_size=50,  )
ReplayBuffer.push_and_pop(self, data,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\dcgan\dcgan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\discogan\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root,  transforms_=None, mode='train',  )
ImageDataset.__len__(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\discogan\discogan.py
----------------methods----------------

---------------functions---------------
sample_images(batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\discogan\models.py
----------------methods----------------
Discriminator.__init__(self, input_shape,   )
Discriminator.forward(self, img,   )
GeneratorUNet.__init__(self, input_shape,   )
GeneratorUNet.forward(self, x,   )
UNetDown.__init__(self, in_size, out_size,  normalize=True, dropout=0.0,  )
UNetDown.forward(self, x,   )
UNetUp.__init__(self, in_size, out_size,  dropout=0.0,  )
UNetUp.forward(self, x, skip_input,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\dragan\dragan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, noise,   )

---------------functions---------------
compute_gradient_penalty(D, X,   )
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\dualgan\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root,  transforms_=None, mode="train",  )
ImageDataset.__len__(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\dualgan\dualgan.py
----------------methods----------------

---------------functions---------------
compute_gradient_penalty(D, real_samples, fake_samples,   )
sample_images(batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\dualgan\models.py
----------------methods----------------
Discriminator.__init__(self,  in_channels=3,  )
Discriminator.forward(self, img,   )
Generator.__init__(self,  channels=3,  )
Generator.forward(self, x,   )
UNetDown.__init__(self, in_size, out_size,  normalize=True, dropout=0.0,  )
UNetDown.forward(self, x,   )
UNetUp.__init__(self, in_size, out_size,  dropout=0.0,  )
UNetUp.forward(self, x, skip_input,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\ebgan\ebgan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, noise,   )

---------------functions---------------
pullaway_loss(embeddings,   )
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\esrgan\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root, hr_shape,   )
ImageDataset.__len__(self,   )

---------------functions---------------
denormalize(tensors,   )


mlmodels\model_tch\raw\pytorch_gan\models\esrgan\esrgan.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\esrgan\models.py
----------------methods----------------
DenseResidualBlock.__init__(self, filters,  res_scale=0.2,  )
DenseResidualBlock.forward(self, x,   )
Discriminator.__init__(self, input_shape,   )
Discriminator.forward(self, img,   )
FeatureExtractor.__init__(self,   )
FeatureExtractor.forward(self, img,   )
GeneratorRRDB.__init__(self, channels,  filters=64, num_res_blocks=16, num_upsample=2,  )
GeneratorRRDB.forward(self, x,   )
ResidualInResidualDenseBlock.__init__(self, filters,  res_scale=0.2,  )
ResidualInResidualDenseBlock.forward(self, x,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\esrgan\test_on_image.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\gan\gan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\infogan\infogan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, noise, labels, code,   )

---------------functions---------------
sample_image(n_row, batches_done,   )
to_categorical(y, num_columns,   )
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\lsgan\lsgan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\munit\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root,  transforms_=None, mode="train",  )
ImageDataset.__len__(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\munit\models.py
----------------methods----------------
AdaptiveInstanceNorm2d.__init__(self, num_features,  eps=1e-5, momentum=0.1,  )
AdaptiveInstanceNorm2d.__repr__(self,   )
AdaptiveInstanceNorm2d.forward(self, x,   )
ContentEncoder.__init__(self,  in_channels=3, dim=64, n_residual=3, n_downsample=2,  )
ContentEncoder.forward(self, x,   )
Decoder.__init__(self,  out_channels=3, dim=64, n_residual=3, n_upsample=2, style_dim=8,  )
Decoder.assign_adain_params(self, adain_params,   )
Decoder.forward(self, content_code, style_code,   )
Decoder.get_num_adain_params(self,   )
Encoder.__init__(self,  in_channels=3, dim=64, n_residual=3, n_downsample=2, style_dim=8,  )
Encoder.forward(self, x,   )
LambdaLR.__init__(self, n_epochs, offset, decay_start_epoch,   )
LambdaLR.step(self, epoch,   )
LayerNorm.__init__(self, num_features,  eps=1e-5, affine=True,  )
LayerNorm.forward(self, x,   )
MLP.__init__(self, input_dim, output_dim,  dim=256, n_blk=3, activ="relu",  )
MLP.forward(self, x,   )
MultiDiscriminator.__init__(self,  in_channels=3,  )
MultiDiscriminator.compute_loss(self, x, gt,   )
MultiDiscriminator.forward(self, x,   )
ResidualBlock.__init__(self, features,  norm="in",  )
ResidualBlock.forward(self, x,   )
StyleEncoder.__init__(self,  in_channels=3, dim=64, n_downsample=2, style_dim=8,  )
StyleEncoder.forward(self, x,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\munit\munit.py
----------------methods----------------

---------------functions---------------
sample_images(batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\pix2pix\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root,  transforms_=None, mode="train",  )
ImageDataset.__len__(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\pix2pix\models.py
----------------methods----------------
Discriminator.__init__(self,  in_channels=3,  )
Discriminator.forward(self, img_A, img_B,   )
GeneratorUNet.__init__(self,  in_channels=3, out_channels=3,  )
GeneratorUNet.forward(self, x,   )
UNetDown.__init__(self, in_size, out_size,  normalize=True, dropout=0.0,  )
UNetDown.forward(self, x,   )
UNetUp.__init__(self, in_size, out_size,  dropout=0.0,  )
UNetUp.forward(self, x, skip_input,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\pix2pix\pix2pix.py
----------------methods----------------

---------------functions---------------
sample_images(batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\pixelda\mnistm.py
----------------methods----------------
MNISTM.__getitem__(self, index,   )
MNISTM.__init__(self, root,  mnist_root="data", train=True, transform=None, target_transform=None, download=False,  )
MNISTM.__len__(self,   )
MNISTM._check_exists(self,   )
MNISTM.download(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\pixelda\pixelda.py
----------------methods----------------
Classifier.__init__(self,   )
Classifier.forward(self, img,   )
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, img, z,   )
ResidualBlock.__init__(self,  in_features=64, out_features=64,  )
ResidualBlock.forward(self, x,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\relativistic_gan\relativistic_gan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\sgan\sgan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, noise,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\softmax_gan\softmax_gan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------
log(x,   )


mlmodels\model_tch\raw\pytorch_gan\models\srgan\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root, hr_shape,   )
ImageDataset.__len__(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\srgan\models.py
----------------methods----------------
Discriminator.__init__(self, input_shape,   )
Discriminator.forward(self, img,   )
FeatureExtractor.__init__(self,   )
FeatureExtractor.forward(self, img,   )
GeneratorResNet.__init__(self,  in_channels=3, out_channels=3, n_residual_blocks=16,  )
GeneratorResNet.forward(self, x,   )
ResidualBlock.__init__(self, in_features,   )
ResidualBlock.forward(self, x,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\srgan\srgan.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\stargan\datasets.py
----------------methods----------------
CelebADataset.__getitem__(self, index,   )
CelebADataset.__init__(self, root,  transforms_=None, mode="train", attributes=None,  )
CelebADataset.__len__(self,   )
CelebADataset.get_annotations(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\stargan\models.py
----------------methods----------------
Discriminator.__init__(self,  img_shape=(3, 128, 128), c_dim=5, n_strided=6,  )
Discriminator.forward(self, img,   )
GeneratorResNet.__init__(self,  img_shape=(3, 128, 128), res_blocks=9, c_dim=5,  )
GeneratorResNet.forward(self, x, c,   )
ResidualBlock.__init__(self, in_features,   )
ResidualBlock.forward(self, x,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\stargan\stargan.py
----------------methods----------------

---------------functions---------------
compute_gradient_penalty(D, real_samples, fake_samples,   )
criterion_cls(logit, target,   )
sample_images(batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\unit\datasets.py
----------------methods----------------
ImageDataset.__getitem__(self, index,   )
ImageDataset.__init__(self, root,  transforms_=None, unaligned=False, mode="train",  )
ImageDataset.__len__(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\unit\models.py
----------------methods----------------
Discriminator.__init__(self, input_shape,   )
Discriminator.forward(self, img,   )
Encoder.__init__(self,  in_channels=3, dim=64, n_downsample=2, shared_block=None,  )
Encoder.forward(self, x,   )
Encoder.reparameterization(self, mu,   )
Generator.__init__(self,  out_channels=3, dim=64, n_upsample=2, shared_block=None,  )
Generator.forward(self, x,   )
LambdaLR.__init__(self, n_epochs, offset, decay_start_epoch,   )
LambdaLR.step(self, epoch,   )
ResidualBlock.__init__(self, features,   )
ResidualBlock.forward(self, x,   )

---------------functions---------------
weights_init_normal(m,   )


mlmodels\model_tch\raw\pytorch_gan\models\unit\unit.py
----------------methods----------------

---------------functions---------------
compute_kl(mu,   )
sample_images(batches_done,   )


mlmodels\model_tch\raw\pytorch_gan\models\wgan\wgan.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\wgan_div\wgan_div.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_gan\models\wgan_gp\wgan_gp.py
----------------methods----------------
Discriminator.__init__(self,   )
Discriminator.forward(self, img,   )
Generator.__init__(self,   )
Generator.forward(self, z,   )

---------------functions---------------
compute_gradient_penalty(D, real_samples, fake_samples,   )


mlmodels\model_tch\raw\pytorch_vae\experiment.py
----------------methods----------------
VAEXperiment.__init__(self, vae_model,   )
VAEXperiment.configure_optimizers(self,   )
VAEXperiment.data_transforms(self,   )
VAEXperiment.sample_images(self,   )
VAEXperiment.train_dataloader(self,   )
VAEXperiment.val_dataloader(self,   )
VAEXperiment.validation_end(self, outputs,   )
VAEXperiment.validation_step(self, batch, batch_idx,  optimizer_idx=0,  )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\run.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\utils.py
----------------methods----------------

---------------functions---------------
data_loader(fn,   )


mlmodels\model_tch\raw\pytorch_vae\models\base.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\betatc_vae.py
----------------methods----------------
BetaTCVAE.__init__(self, in_channels,   )
BetaTCVAE.encode(self, input,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\beta_vae.py
----------------methods----------------
BetaVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\cat_vae.py
----------------methods----------------
CategoricalVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\cvae.py
----------------methods----------------
ConditionalVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\dfcvae.py
----------------methods----------------
DFCVAE.__init__(self, in_channels,   )
DFCVAE.encode(self, input,   )
DFCVAE.loss_function(self,   *args, **kwargs)

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\dip_vae.py
----------------methods----------------
DIPVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\fvae.py
----------------methods----------------
FactorVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\gamma_vae.py
----------------methods----------------
GammaVAE.I_function(self, a, b, c, d,   )
GammaVAE.__init__(self, in_channels,   )
GammaVAE.encode(self, input,   )
GammaVAE.loss_function(self,   *args, **kwargs)
GammaVAE.vae_gamma_kl_loss(self, a, b, c, d,   )
GammaVAE.weight_init(self,   )

---------------functions---------------
init_(m,   )


mlmodels\model_tch\raw\pytorch_vae\models\hvae.py
----------------methods----------------
HVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\info_vae.py
----------------methods----------------
InfoVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\iwae.py
----------------methods----------------
IWAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\joint_vae.py
----------------methods----------------
JointVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\logcosh_vae.py
----------------methods----------------
LogCoshVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\lvae.py
----------------methods----------------
EncoderBlock.__init__(self, in_channels,   )
LVAE.__init__(self, in_channels,   )
LVAE.encode(self, input,   )
LVAE.merge_gauss(self, mu_1,   )
LadderBlock.__init__(self, in_channels,   )

---------------functions---------------
conv_out_shape(img_size,   )


mlmodels\model_tch\raw\pytorch_vae\models\miwae.py
----------------methods----------------
MIWAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\mssim_vae.py
----------------methods----------------
MSSIM.__init__(self, in_channels,   )
MSSIM.ssim(self, img1,   )
MSSIMVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\swae.py
----------------methods----------------
SWAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\twostage_vae.py
----------------methods----------------
TwoStageVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\types_.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\vampvae.py
----------------methods----------------
VampVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\vanilla_vae.py
----------------methods----------------
VanillaVAE.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\vq_vae.py
----------------methods----------------
ResidualLayer.__init__(self, in_channels,   )
VQVAE.__init__(self, in_channels,   )
VectorQuantizer.__init__(self, num_embeddings,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\wae_mmd.py
----------------methods----------------
WAE_MMD.__init__(self, in_channels,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\models\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\bvae.py
----------------methods----------------
TestVAE.setUp(self,   )
TestVAE.test_forward(self,   )
TestVAE.test_loss(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_betatcvae.py
----------------methods----------------
TestBetaTCVAE.setUp(self,   )
TestBetaTCVAE.test_forward(self,   )
TestBetaTCVAE.test_generate(self,   )
TestBetaTCVAE.test_loss(self,   )
TestBetaTCVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_cat_vae.py
----------------methods----------------
TestVAE.setUp(self,   )
TestVAE.test_forward(self,   )
TestVAE.test_loss(self,   )
TestVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_dfc.py
----------------methods----------------
TestDFCVAE.setUp(self,   )
TestDFCVAE.test_forward(self,   )
TestDFCVAE.test_loss(self,   )
TestDFCVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_dipvae.py
----------------methods----------------
TestDIPVAE.setUp(self,   )
TestDIPVAE.test_forward(self,   )
TestDIPVAE.test_generate(self,   )
TestDIPVAE.test_loss(self,   )
TestDIPVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_fvae.py
----------------methods----------------
TestFAE.setUp(self,   )
TestFAE.test_forward(self,   )
TestFAE.test_loss(self,   )
TestFAE.test_optim(self,   )
TestFAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_gvae.py
----------------methods----------------
TestGammaVAE.setUp(self,   )
TestGammaVAE.test_forward(self,   )
TestGammaVAE.test_loss(self,   )
TestGammaVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_hvae.py
----------------methods----------------
TestHVAE.setUp(self,   )
TestHVAE.test_forward(self,   )
TestHVAE.test_loss(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_iwae.py
----------------methods----------------
TestIWAE.setUp(self,   )
TestIWAE.test_forward(self,   )
TestIWAE.test_loss(self,   )
TestIWAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_joint_Vae.py
----------------methods----------------
TestVAE.setUp(self,   )
TestVAE.test_forward(self,   )
TestVAE.test_loss(self,   )
TestVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_logcosh.py
----------------methods----------------
TestVAE.setUp(self,   )
TestVAE.test_forward(self,   )
TestVAE.test_loss(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_lvae.py
----------------methods----------------
TestLVAE.setUp(self,   )
TestLVAE.test_forward(self,   )
TestLVAE.test_loss(self,   )
TestLVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_miwae.py
----------------methods----------------
TestMIWAE.setUp(self,   )
TestMIWAE.test_forward(self,   )
TestMIWAE.test_generate(self,   )
TestMIWAE.test_loss(self,   )
TestMIWAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_mssimvae.py
----------------methods----------------
TestMSSIMVAE.setUp(self,   )
TestMSSIMVAE.test_forward(self,   )
TestMSSIMVAE.test_loss(self,   )
TestMSSIMVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_swae.py
----------------methods----------------
TestSWAE.setUp(self,   )
TestSWAE.test_forward(self,   )
TestSWAE.test_loss(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_vae.py
----------------methods----------------
TestVAE.setUp(self,   )
TestVAE.test_forward(self,   )
TestVAE.test_loss(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_vq_vae.py
----------------methods----------------
TestVQVAE.setUp(self,   )
TestVQVAE.test_forward(self,   )
TestVQVAE.test_generate(self,   )
TestVQVAE.test_loss(self,   )
TestVQVAE.test_sample(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\test_wae.py
----------------methods----------------
TestWAE.setUp(self,   )
TestWAE.test_forward(self,   )
TestWAE.test_loss(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\text_cvae.py
----------------methods----------------
TestCVAE.setUp(self,   )
TestCVAE.test_loss(self,   )

---------------functions---------------


mlmodels\model_tch\raw\pytorch_vae\tests\text_vamp.py
----------------methods----------------
TestVVAE.setUp(self,   )
TestVVAE.test_forward(self,   )
TestVVAE.test_loss(self,   )

---------------functions---------------


mlmodels\model_tch\raw\textcnn\dataset.py
----------------methods----------------

---------------functions---------------
clean_str(string,   )
create_data_iterator(tr_batch_size, val_batch_size, tabular_train, tabular_valid, d,   )
create_tabular_dataset(path_train, path_valid,  lang='en', pretrained_emb='glove.6B.300d',  )
split_train_valid(path_data, path_train, path_valid,  frac=0.7,  )


mlmodels\model_tch\raw\textcnn\main.py
----------------methods----------------

---------------functions---------------
main(  )


mlmodels\model_tch\raw\textcnn\model.py
----------------methods----------------
textCNN.__init__(self, vocab_built, dim_channel, kernel_wins, dropout_rate, num_class,   )
textCNN.forward(self, x,   )

---------------functions---------------


mlmodels\model_tch\raw\textcnn\training.py
----------------methods----------------

---------------functions---------------
train(m, device, train_itr, optimizer, epoch, max_epoch,   )
valid(m, device, test_itr,   )


mlmodels\model_tch\raw\vae\cli_generate_data.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae\network_test.py
----------------methods----------------
BetaVAE_H.__init__(self,  z_dim=10, nc=3,  )
BetaVAE_H._decode(self, z,   )
BetaVAE_H._encode(self, x,   )
BetaVAE_H.forward(self, x,   )
BetaVAE_H.weight_init(self,   )
BetaVAE_new.__init__(self,  z_dim=10, nc=3,  )
BetaVAE_new._decode(self, z,   )
BetaVAE_new._encode(self, x,   )
BetaVAE_new.forward(self, x,   )
BetaVAE_new.weight_init(self,   )
View.__init__(self, size,   )
View.forward(self, tensor,   )

---------------functions---------------
kaiming_init(m,   )
normal_init(m, mean, std,   )
reparametrize(mu, logvar,   )


mlmodels\model_tch\raw\vae\util.py
----------------methods----------------

---------------functions---------------
create_sin_2d_array_cv(x, y,  resoltuion=data['resolution'], amp=data['amplitude'],  )
create_sin_2d_array_plt(x, y,  xmax=data['resolution'], ymax=data['resolution'], amp=data['amplitude'],  )
generate_random_cos( n_rand_starts=1, a=1, w=1, b=0, c=0, x_upbound=1, x_downbound=-1, step=0.2,  )
generate_random_sin( n_rand_starts=100, amplitude=1, n_pis=4, omega=1, step=0.2,  )
generate_train_img(folder,  N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2, cmax=2, step=0.1, wfreq=0.5,  )
generate_train_img_cv(folder,  N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2, cmax=2, step=0.1, wfreq=0.5,  )
generate_train_npz(folder,  N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2, cmax=2, step=0.1, wfreq=0.5, resoltuion=data['resolution'],  )
generate_train_npz_cv(folder,  N_type=1, amax=5, wmin=5, wmax=10, bmin=-2, bmax=2, cmin=-2, cmax=2, step=0.1, wfreq=0.5, resoltuion=data['resolution'],  )
get_resolution(  )
plot_save_disk(x, y, filename,  xmax=data['resolution'], ymax=data['resolution'], amp=data['amplitude'],  )
plot_save_disk_cv(x, y, filename,  xmax=data['resolution'], ymax=data['resolution'], amp=data['amplitude'],  )
set_resolution( resolution=64,  )


mlmodels\model_tch\raw\vae\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae\models\cli.py
----------------methods----------------

---------------functions---------------
load_argument(  )


mlmodels\model_tch\raw\vae\models\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae\models\Beta_VAE\dataset.py
----------------methods----------------
CustomImageFolder.__getitem__(self, index,   )
CustomImageFolder.__init__(self, root,  transform=None,  )
CustomTensorDataset.__getitem__(self, index,   )
CustomTensorDataset.__init__(self, data_tensor,   )
CustomTensorDataset.__len__(self,   )

---------------functions---------------
is_power_of_2(num,   )
return_data(args,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE\main.py
----------------methods----------------

---------------functions---------------
main(args,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE\model.py
----------------methods----------------
BetaVAE_B.__init__(self,  z_dim=10, nc=1,  )
BetaVAE_B._decode(self, z,   )
BetaVAE_B._encode(self, x,   )
BetaVAE_B.forward(self, x,   )
BetaVAE_B.weight_init(self,   )
BetaVAE_H.__init__(self,  z_dim=10, nc=3,  )
BetaVAE_H._decode(self, z,   )
BetaVAE_H._encode(self, x,   )
BetaVAE_H.forward(self, x,   )
BetaVAE_H.weight_init(self,   )
BetaVAE_H_4_nn.__init__(self,  z_dim=10, nc=3,  )
BetaVAE_H_4_nn._decode(self, z,   )
BetaVAE_H_4_nn._encode(self, x,   )
BetaVAE_H_4_nn.forward(self, x,   )
BetaVAE_H_4_nn.weight_init(self,   )
View.__init__(self, size,   )
View.forward(self, tensor,   )

---------------functions---------------
kaiming_init(m,   )
normal_init(m, mean, std,   )
reparametrize(mu, logvar,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE\solver.py
----------------methods----------------
DataGather.__init__(self,   )
DataGather.flush(self,   )
DataGather.get_empty_data_dict(self,   )
DataGather.insert(self,   **kwargs)
Solver.__init__(self, args,   )
Solver.load_checkpoint(self, filename,   )
Solver.net_mode(self, train,   )
Solver.save_checkpoint(self, filename,  silent=True,  )
Solver.train(self,   )
Solver.viz_lines(self,   )
Solver.viz_reconstruction(self,   )
Solver.viz_traverse(self,  limit=3, inter=2/3, loc=-1,  )

---------------functions---------------
kl_divergence(mu, logvar,   )
reconstruction_loss(x, x_recon, distribution,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE\utils.py
----------------methods----------------

---------------functions---------------
cuda(tensor, uses_cuda,   )
grid2gif(image_str, output_gif,  delay=100,  )
str2bool(v,   )
where(cond, x, y,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae\models\Beta_VAE_fft\dataset.py
----------------methods----------------
CustomImageFolder.__getitem__(self, index,   )
CustomImageFolder.__init__(self, root,  transform=None,  )
CustomTensorDataset.__getitem__(self, index,   )
CustomTensorDataset.__init__(self, data_tensor,   )
CustomTensorDataset.__len__(self,   )

---------------functions---------------
is_power_of_2(num,   )
return_data(args,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE_fft\main.py
----------------methods----------------

---------------functions---------------
main(args,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE_fft\model.py
----------------methods----------------
BetaVAE_B.__init__(self,  z_dim=10, nc=1,  )
BetaVAE_B._decode(self, z,   )
BetaVAE_B._encode(self, x,   )
BetaVAE_B.forward(self, x,   )
BetaVAE_B.weight_init(self,   )
BetaVAE_H.__init__(self,  z_dim=10, nc=3,  )
BetaVAE_H._decode(self, z,   )
BetaVAE_H._encode(self, x,   )
BetaVAE_H.forward(self, x,   )
BetaVAE_H.weight_init(self,   )
BetaVAE_fft.__init__(self,  z_dim=10, nc=3,  )
BetaVAE_fft._decode(self, z,   )
BetaVAE_fft._encode(self, x,   )
BetaVAE_fft.forward(self, x,   )
BetaVAE_fft.weight_init(self,   )
View.__init__(self, size,   )
View.forward(self, tensor,   )

---------------functions---------------
kaiming_init(m,   )
normal_init(m, mean, std,   )
reparametrize(mu, logvar,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE_fft\solver.py
----------------methods----------------
DataGather.__init__(self,   )
DataGather.flush(self,   )
DataGather.get_empty_data_dict(self,   )
DataGather.insert(self,   **kwargs)
Solver.__init__(self, args,   )
Solver.load_checkpoint(self, filename,   )
Solver.net_mode(self, train,   )
Solver.save_checkpoint(self, filename,  silent=True,  )
Solver.train(self,   )
Solver.viz_lines(self,   )
Solver.viz_reconstruction(self,   )
Solver.viz_traverse(self,  limit=3, inter=2/3, loc=-1,  )

---------------functions---------------
kl_divergence(mu, logvar,   )
reconstruction_loss(x, x_recon, distribution,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE_fft\utils.py
----------------methods----------------

---------------functions---------------
cuda(tensor, uses_cuda,   )
grid2gif(image_str, output_gif,  delay=100,  )
str2bool(v,   )
where(cond, x, y,   )


mlmodels\model_tch\raw\vae\models\Beta_VAE_fft\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\exp_utils.py
----------------methods----------------

---------------functions---------------
create_exp_dir(dir_path,  scripts_to_save=None, debug=False,  )
get_logger(log_path,   **kwargs)
logging(s, log_path,  print_=True, log_=True,  )
save_checkpoint(model, optimizer, path, epoch,   )


mlmodels\model_tch\raw\vae_pretraining_encoder\lm.py
----------------methods----------------

---------------functions---------------
init_config(  )
main(args,   )
test(model, test_data_batch, args,   )


mlmodels\model_tch\raw\vae_pretraining_encoder\prepare_data.py
----------------methods----------------

---------------functions---------------
download_file_from_google_drive(id, destination,   )
get_confirm_token(response,   )
save_response_content(response, destination,   )


mlmodels\model_tch\raw\vae_pretraining_encoder\text_anneal_fb.py
----------------methods----------------

---------------functions---------------
init_config(  )
main(args,   )
test(model, test_data_batch, mode, args,  verbose=True,  )


mlmodels\model_tch\raw\vae_pretraining_encoder\text_beta.py
----------------methods----------------

---------------functions---------------
init_config(  )
main(args,   )
test(model, test_data_batch, mode, args,  verbose=True,  )


mlmodels\model_tch\raw\vae_pretraining_encoder\text_get_mean.py
----------------methods----------------

---------------functions---------------
init_config(  )
main(args,   )
save_latents(args, vae, test_data_batch, test_label_batch, str_,   )
test(model, test_data_batch, mode, args,  verbose=True,  )


mlmodels\model_tch\raw\vae_pretraining_encoder\text_ss_ft.py
----------------methods----------------

---------------functions---------------
init_config(  )
main(args,   )
test(model, test_data_batch, test_labels_batch, mode, args,  verbose=True,  )


mlmodels\model_tch\raw\vae_pretraining_encoder\utils.py
----------------methods----------------
xavier_normal_initializer.__call__(self, tensor,   )

---------------functions---------------
calc_au(model, test_data_batch,  delta=0.01,  )
calc_iwnll(model, test_data_batch, args,  ns=100,  )
calc_mi(model, test_data_batch,   )
call_multi_bleu_perl(fname_bleu_script, fname_hyp, fname_ref,  verbose=True,  )
reconstruct(model, test_data_batch, vocab, strategy, fname,   )
sample_sentences(vae, vocab, device, num_sentences,   )
visualize_latent(args, epoch, vae, device, test_data,   )


mlmodels\model_tch\raw\vae_pretraining_encoder\config\config_ptb.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\config\config_short_yelp.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\config\config_snli.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\config\config_yahoo.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\config\config_yahoo_label.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\data\text_data.py
----------------methods----------------
MonoTextData.__init__(self, fname,  label=False, max_length=None, vocab=None,  )
MonoTextData.__len__(self,   )
MonoTextData._read_corpus(self, fname, label, max_length, vocab,   )
MonoTextData._to_tensor(self, batch_data, batch_first, device,   )
MonoTextData.create_data_batch(self, batch_size, device,  batch_first=False,  )
MonoTextData.create_data_batch_labels(self, batch_size, device,  batch_first=False,  )
MonoTextData.data_iter(self, batch_size, device,  batch_first=False, shuffle=True,  )
MonoTextData.data_sample(self, nsample, device,  batch_first=False, shuffle=True,  )
VocabEntry.__contains__(self, word,   )
VocabEntry.__getitem__(self, word,   )
VocabEntry.__init__(self,  word2id=None,  )
VocabEntry.__len__(self,   )
VocabEntry.add(self, word,   )
VocabEntry.decode_sentence(self, sentence,   )
VocabEntry.from_corpus(fname,   )
VocabEntry.id2word(self, wid,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\data\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\utils.py
----------------methods----------------

---------------functions---------------
generate_grid(zmin, zmax, dz, device,  ndim=2,  )
log_sum_exp(value,  dim=None, keepdim=False,  )
safe_log(z,   )


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\vae.py
----------------methods----------------
VAE.KL(self, x,   )
VAE.__init__(self, encoder, decoder, args,   )
VAE.calc_infer_mean(self, x,   )
VAE.calc_mi_q(self, x,   )
VAE.calc_model_posterior_mean(self, x, grid_z,   )
VAE.decode(self, z, strategy,  K=10,  )
VAE.encode(self, x,  nsamples=1,  )
VAE.encode_stats(self, x,   )
VAE.eval_complete_ll(self, x, z,   )
VAE.eval_cond_ll(self, x, z,   )
VAE.eval_inference_dist(self, x, z,  param=None,  )
VAE.eval_log_model_posterior(self, x, grid_z,   )
VAE.eval_prior_dist(self, zrange,   )
VAE.loss(self, x, kl_weight,  nsamples=1,  )
VAE.loss_iw(self, x, kl_weight,  nsamples=50, ns=10,  )
VAE.nll_iw(self, x, nsamples,  ns=100,  )
VAE.reconstruct(self, x,  decoding_strategy="greedy", K=5,  )
VAE.sample_from_inference(self, x,  nsamples=1,  )
VAE.sample_from_posterior(self, x, nsamples,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\decoders\decoder.py
----------------methods----------------
DecoderBase.__init__(self,   )
DecoderBase.beam_search_decode(self, z, K,   )
DecoderBase.decode(self, x, z,   )
DecoderBase.freeze(self,   )
DecoderBase.greedy_decode(self, z,   )
DecoderBase.log_probability(self, x, z,   )
DecoderBase.reconstruct_error(self, x, z,   )
DecoderBase.sample_decode(self, z,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\decoders\decoder_help.py
----------------methods----------------
BeamSearchNode.__init__(self, hiddenstate, previousNode, wordId, logProb, length,   )
BeamSearchNode.eval(self,  alpha=1.0,  )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\decoders\decoder_helper.py
----------------methods----------------
BeamSearchNode.__init__(self, hiddenstate, previousNode, wordId, logProb, length,   )
BeamSearchNode.eval(self,  alpha=1.0,  )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\decoders\dec_lstm.py
----------------methods----------------
LSTMDecoder.__init__(self, args, vocab, model_init, emb_init,   )
LSTMDecoder.beam_search_decode(self, z,  K=5,  )
LSTMDecoder.decode(self, input, z,   )
LSTMDecoder.greedy_decode(self, z,   )
LSTMDecoder.log_probability(self, x, z,   )
LSTMDecoder.reconstruct_error(self, x, z,   )
LSTMDecoder.reset_parameters(self, model_init, emb_init,   )
LSTMDecoder.sample_decode(self, z,  greedy=False,  )
LSTMDecoder.sample_text(self, input, z, EOS, device,   )
VarLSTMDecoder.__init__(self, args, vocab, model_init, emb_init,   )
VarLSTMDecoder.decode(self, input, z,   )
VarLSTMDecoder.reconstruct_error(self, x, z,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\decoders\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\discriminators\discriminator_linear.py
----------------methods----------------
LinearDiscriminator.__init__(self, args, encoder,   )
LinearDiscriminator.get_performance(self, batch_data, batch_labels,   )
MLPDiscriminator.__init__(self, args, encoder,   )
MLPDiscriminator.get_performance(self, batch_data, batch_labels,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\discriminators\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\encoders\encoder.py
----------------methods----------------
EncoderBase.__init__(self,   )
EncoderBase.calc_mi(self, x,   )
EncoderBase.encode(self, input, nsamples,   )
EncoderBase.eval_inference_dist(self, x, z,  param=None,  )
EncoderBase.forward(self, x,   )
EncoderBase.sample(self, input, nsamples,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\encoders\enc_lstm.py
----------------methods----------------
GaussianLSTMEncoder.__init__(self, args, vocab_size, model_init, emb_init,   )
GaussianLSTMEncoder.forward(self, input,   )
GaussianLSTMEncoder.reset_parameters(self, model_init, emb_init,   )
VarLSTMEncoder.__init__(self, args, vocab_size, model_init, emb_init,   )
VarLSTMEncoder.encode(self, input, nsamples,   )
VarLSTMEncoder.forward(self, input,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\encoders\gaussian_encoder.py
----------------methods----------------
GaussianEncoderBase.__init__(self,   )
GaussianEncoderBase.calc_mi(self, x,   )
GaussianEncoderBase.encode(self, input, nsamples,   )
GaussianEncoderBase.encode_stats(self, x,   )
GaussianEncoderBase.eval_inference_dist(self, x, z,  param=None,  )
GaussianEncoderBase.forward(self, x,   )
GaussianEncoderBase.freeze(self,   )
GaussianEncoderBase.reparameterize(self, mu, logvar,  nsamples=1,  )
GaussianEncoderBase.sample(self, input, nsamples,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\encoders\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\lm\lm_lstm.py
----------------methods----------------
LSTM_LM.__init__(self, args, vocab, model_init, emb_init,   )
LSTM_LM.decode(self, input,   )
LSTM_LM.log_probability(self, x,   )
LSTM_LM.reconstruct_error(self, x,   )
LSTM_LM.reset_parameters(self, model_init, emb_init,   )

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\modules\lm\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\scripts\sampling_training_labels.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\raw\vae_pretraining_encoder\scripts\unsupervised_cluster.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tch\zdocs\transformer_classifier2.py
----------------methods----------------
Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )

---------------functions---------------
_preprocess_XXXX(df,   **kw)
evaluate(model, tokenizer,  prefix="",  )
fit(model,  data_pars=None, model_pars={}, compute_pars=None, out_pars=None,  *args, **kw)
get_dataset( data_pars=None,  **kw)
get_eval_report(labels, preds,   )
get_mismatched(labels, preds,   )
get_params( choice=0, data_path="dataset/",  **kw)
load( out_pars=None,  )
metrics(task_name, preds, labels,   )
metrics_evaluate(  )
path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
predict(model,  sess=None, data_pars=None, out_pars=None, compute_pars=None,  **kw)
reset_model(  )
save(model, out_pars,   )
test( data_path="dataset/", pars_choice=0,  )


mlmodels\model_tf\1_lstm.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
fit_metrics(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  )
get_dataset( data_pars=None,  )
get_params( param_pars={},  **kw)
load( load_pars=None,  )
metrics(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None, get_hidden_state=False, init_value=None,  )
reset_model(  )
save(model,  session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="test01", config_mode="test",  )


mlmodels\model_tf\temporal_fusion_google.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kwarg)
fit_metrics(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  )
get_dataset( data_pars=None,  )
get_params( param_pars={},  **kw)
load( load_pars=None,  )
metrics(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None, get_hidden_state=False, init_value=None,  )
reset_model(  )
save(model,  session=None, save_pars=None,  )
test( data_path="dataset/", pars_choice="test01", config_mode="test",  )


mlmodels\model_tf\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )
os_file_path(data_path,   )
os_module_path(  )
os_package_root_path(filepath,  sublevel=0, path_add="",  )
set_root_dir(  )


mlmodels\model_tf\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\10_encoder_vanilla.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )
Model.build_model(self,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,   )
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\11_bidirectional_vanilla.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value_forward=None, init_value_backward=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\12_vanilla_2path.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value_forward=None, init_value_backward=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\13_lstm_seq2seq.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\14_lstm_attention.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\15_lstm_seq2seq_attention.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\16_lstm_seq2seq_bidirectional.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value_forward=None, init_value_backward=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\17_lstm_seq2seq_bidirectional_attention.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value_forward=None, init_value_backward=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\18_lstm_attention_scaleddot.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, seq_len,  forget_bias=0.1, epoch=500,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\19_lstm_dilated.py
----------------methods----------------
Model.__init__(self, steps, dimension_input, dimension_output,  learning_rate=0.001, hidden_structs=[20], dilations=[1, 1, 1, 1], epoch=500,  )
Model.anchor(signal, weight,   )

---------------functions---------------
contruct_cells(hidden_structs,   )
dilated_rnn(cell, inputs, rate, states,  scope="default",  )
fit(model, data_frame,   )
multi_dilated_rnn(cells, inputs, dilations, states,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
rnn_reformat(x, input_dims, n_steps,   )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\20_only_attention.py
----------------methods----------------
Model.__init__(self, seq_len, learning_rate, dimension_input, dimension_output,  epoch=100,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,   )
sinusoidal_positional_encoding(inputs, num_units,  zero_pad=False, scale=False,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\21_multihead_attention.py
----------------methods----------------
Model.__init__(self, dimension_input, dimension_output, seq_len, learning_rate,  num_heads=8, attn_windows=range(1, 6), epoch=1,  )
Model.anchor(signal, weight,   )
Model.multihead_attn(self, inputs, masks,   )
Model.window_mask(self, h_w,   )

---------------functions---------------
embed_seq(inputs,  vocab_size=None, embed_dim=None, zero_pad=False, scale=False,  )
fit(model, data_frame,   )
layer_norm(inputs,  epsilon=1e-8,  )
learned_positional_encoding(inputs, embed_dim,  zero_pad=False, scale=False,  )
pointwise_feedforward(inputs,  num_units=[None, None], activation=None,  )
predict(model, sess, data_frame,   )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\22_lstm_bahdanau.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10, epoch=100, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\23_lstm_luong.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10, epoch=100, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\24_lstm_luong_bahdanau.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10, epoch=1, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\25_dnc.py
----------------methods----------------
Model.__init__(self, learning_rate, size, size_layer, output_size, epoch, timestep, access_config, controller_config, clip_value,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\26_lstm_residual.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  epoch=1, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\27_byte_net.py
----------------methods----------------
Model.__init__(self, size, output_size, channels, encoder_dilations, encoder_filter_width,  learning_rate=0.001, beta1=0.5, epoch=1, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width,  causal=True,  )
conv1d(input_, output_channels,  dilation=1, filter_width=1, causal=False,  )
fit(model, data_frame,   )
layer_normalization(x,  epsilon=1e-8,  )
predict(model, sess, data_frame,   )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\28_attention_is_all_you_need.py
----------------methods----------------
Model.__init__(self, size_layer, embedded_size, learning_rate, size, output_size,  num_blocks=2, num_heads=8, min_freq=50, epoch=1, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
label_smoothing(inputs,  epsilon=0.1,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_position_encoding(inputs, mask, embed_dim,   )
multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads,   )
pointwise_feedforward(inputs, hidden_units,  activation=None,  )
predict(model, sess, data_frame,   )
sinusoidal_position_encoding(inputs, mask, repr_dim,   )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\29_fairseq.py
----------------methods----------------
Model.__init__(self, n_layers, size, output_size, emb_size, n_hidden, n_attn_heads, learning_rate,  epoch=1, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
decoder_block(inp, n_hidden, filter_size,   )
encoder_block(inp, n_hidden, filter_size,   )
fit(model, data_frame,   )
glu(x,   )
layer(inp, conv_block, kernel_width, n_hidden,  residual=None,  )
predict(model, sess, data_frame,   )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\2_encoder_lstm.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=5, timestep=5,  )
Model.anchor(signal, weight,   )
Model.build_model(self,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,   )
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20, sess=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\3_bidirectional_lstm.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value_forward=None, init_value_backward=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\4_lstm_2path.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, timestep=5, epoch=10,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value_forward=None, init_value_backward=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\50lstm attention.py
----------------methods----------------
AttentionModel.__init__(self, x, y, layer_1_rnn_units,  attn_dense_nodes=0, epochs=100, batch_size=128, shared_attention_layer=True, chg_yield=False, float_type='float32', regularization=(0.00001, '00001'), window=52, predict=1,  )
AttentionModel.build_attention_rnn(self,   )
AttentionModel.calculate_attentions(self, x_data,   )
AttentionModel.delete_model(self,   )
AttentionModel.fit_model(self,   )
AttentionModel.heatmap(self, data,  title_supplement=None,  )
AttentionModel.load_model(self,   )
AttentionModel.make_shared_layers(self,   )
AttentionModel.save_model(self,   )
AttentionModel.set_learning(self, learning,   )

---------------functions---------------
softmax_activation(x,   )


mlmodels\model_tf\raw\5_gru.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=1, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\6_encoder_gru.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, timestep=5, epoch=1,  )
Model.anchor(signal, weight,   )
Model.build_model(self,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,   )
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\7_bidirectional_gru.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value_forward=None, init_value_backward=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\8_gru_2path.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value_forward=None, init_value_backward=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\9_vanilla.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, epoch=500, timestep=5,  )
Model.anchor(signal, weight,   )

---------------functions---------------
fit(model, data_frame,   )
predict(model, sess, data_frame,  get_hidden_state=False, init_value=None,  )
test( filename="dataset/GOOG-year.csv",  )


mlmodels\model_tf\raw\access.py
----------------methods----------------
MemoryAccess.__init__(self,  memory_size=128, word_size=20, num_reads=1, num_writes=1, name="memory_access",  )
MemoryAccess._build(self, inputs, prev_state,   )
MemoryAccess._read_inputs(self, inputs,   )
MemoryAccess._read_weights(self, inputs, memory, prev_read_weights, link,   )
MemoryAccess._write_weights(self, inputs, memory, usage,   )
MemoryAccess.output_size(self,   )
MemoryAccess.state_size(self,   )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_tf\raw\addressing.py
----------------methods----------------
CosineWeights.__init__(self, num_heads, word_size,  strength_op=tf.nn.softplus, name="cosine_weights",  )
CosineWeights._build(self, memory, keys, strengths,   )
Freeness.__init__(self, memory_size,  name="freeness",  )
Freeness._allocation(self, usage,   )
Freeness._build(self, write_weights, free_gate, read_weights, prev_usage,   )
Freeness._usage_after_read(self, prev_usage, free_gate, read_weights,   )
Freeness._usage_after_write(self, prev_usage, write_weights,   )
Freeness.state_size(self,   )
Freeness.write_allocation_weights(self, usage, write_gates, num_writes,   )
TemporalLinkage.__init__(self, memory_size, num_writes,  name="temporal_linkage",  )
TemporalLinkage._build(self, write_weights, prev_state,   )
TemporalLinkage._link(self, prev_link, prev_precedence_weights, write_weights,   )
TemporalLinkage._precedence_weights(self, prev_precedence_weights, write_weights,   )
TemporalLinkage.directional_read_weights(self, link, prev_read_weights, forward,   )
TemporalLinkage.state_size(self,   )

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_tf\raw\autoencoder.py
----------------methods----------------

---------------functions---------------
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )


mlmodels\model_tf\raw\convert_ipny_cli.py
----------------methods----------------

---------------functions---------------
Run(  )
check(file_list,  dump=False,  )
convert_topython(source_files, data_file, out_dir,   )
scan(data_file,   )


mlmodels\model_tf\raw\dnc.py
----------------methods----------------
DNC.__init__(self, access_config, controller_config, output_size,  clip_value=None, name="dnc",  )
DNC._build(self, inputs, prev_state,   )
DNC._clip_if_enabled(self, x,   )
DNC.initial_state(self, batch_size,  dtype=tf.float32,  )
DNC.output_size(self,   )
DNC.state_size(self,   )

---------------functions---------------


mlmodels\model_tf\raw\deepar\settings.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\deepar\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\deepar\dataset\time_series.py
----------------methods----------------
MockTs.__init__(self,  t_min=0, t_max=30, resolution=.1,  )
MockTs._time_series(t,   )
MockTs.generate_test_data(self, n_steps,   )
MockTs.mock_ts(self,   )
MockTs.next_batch(self, batch_size, n_steps,   )
TimeSeries.__init__(self, pandas_df,  one_hot_root_list=None, grouping_variable='category', scaler=None,  )
TimeSeries._one_hot_padding(self, pandas_df, padding_df,   )
TimeSeries._pad_ts(self, pandas_df, desired_len,  padding_val=0,  )
TimeSeries._sample_ts(pandas_df, desired_len,   )
TimeSeries.next_batch(self, batch_size, n_steps,  target_var='target', verbose=False, padding_value=0,  )

---------------functions---------------


mlmodels\model_tf\raw\deepar\dataset\__init__.py
----------------methods----------------
Dataset.__init__(self,   )
Dataset.next_batch(self,   **kwargs)

---------------functions---------------


mlmodels\model_tf\raw\deepar\model\layers.py
----------------methods----------------
GaussianLayer.__init__(self, output_dim,   **kwargs)
GaussianLayer.build(self, input_shape,   )
GaussianLayer.call(self, x,   )
GaussianLayer.compute_output_shape(self, input_shape,   )

---------------functions---------------


mlmodels\model_tf\raw\deepar\model\loss.py
----------------methods----------------

---------------functions---------------
gaussian_likelihood(sigma,   )


mlmodels\model_tf\raw\deepar\model\lstm.py
----------------methods----------------
DeepAR.__init__(self, ts_obj,  steps_per_epoch=50, epochs=100, loss=gaussian_likelihood, optimizer='adam', with_custom_nn_structure=None,  )
DeepAR.basic_structure(  )
DeepAR.instantiate_and_fit(self,  verbose=False,  )
DeepAR.model(self,   )
DeepAR.predict_theta_from_input(self, input_list,   )

---------------functions---------------
ts_generator(ts_obj, n_steps,   )


mlmodels\model_tf\raw\deepar\model\__init__.py
----------------methods----------------
NNModel.__init__(self,   )
NNModel.instantiate_and_fit(self,   **kwargs)
NNModel.net_structure(self,   **kwargs)

---------------functions---------------


mlmodels\model_tf\raw\deepar\tests\test_dataset.py
----------------methods----------------
TestRecurrentTs.setUp(self,   )
TestRecurrentTs.test_len_padding(self,   )
TestRecurrentTs.test_next_batch_covariates(self,   )
TestRecurrentTs.test_next_batch_production(self,   )
TestRecurrentTs.test_padding_with_one_hot(self,   )
TestRecurrentTs.test_padding_with_one_hot_multiple(self,   )
TestRecurrentTs.test_sample_ts(self,   )
TestRecurrentTs.test_zero_len_padding(self,   )

---------------functions---------------


mlmodels\model_tf\raw\deepar\utils\__init__.py
----------------methods----------------

---------------functions---------------
clear_keras_session(  )
set_seed_and_reset_graph( seed=42,  )


mlmodels\model_tf\raw\temporal_fusion_google\script_download_data.py
----------------methods----------------

---------------functions---------------
download_and_unzip(url, zip_path, csv_path, data_folder,   )
download_electricity(config,   )
download_from_url(url, output_path,   )
download_traffic(config,   )
download_volatility(config,   )
main(expt_name, force_download, output_folder,   )
process_favorita(config,   )
recreate_folder(path,   )
unzip(zip_path, output_file, data_folder,   )


mlmodels\model_tf\raw\temporal_fusion_google\script_hyperparam_opt.py
----------------methods----------------

---------------functions---------------
main(expt_name, use_gpu, restart_opt, model_folder, hyperparam_iterations, data_csv_path, data_formatter,   )


mlmodels\model_tf\raw\temporal_fusion_google\script_train_fixed_params.py
----------------methods----------------

---------------functions---------------
main(expt_name, use_gpu, model_folder, data_csv_path, data_formatter,  use_testing_mode=False,  )


mlmodels\model_tf\raw\temporal_fusion_google\data_formatters\base.py
----------------methods----------------
GenericDataFormatter._check_single_column(input_type,   )
GenericDataFormatter._extract_tuples_from_data_type(data_type, defn,   )
GenericDataFormatter._get_locations(input_types, defn,   )

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\data_formatters\electricity.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\data_formatters\favorita.py
----------------methods----------------
FavoritaFormatter._check_single_column(input_type,   )
FavoritaFormatter.filter_ids(frame,   )

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\data_formatters\traffic.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\data_formatters\volatility.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\data_formatters\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\expt_settings\configs.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\expt_settings\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\libs\hyperparam_opt.py
----------------methods----------------
HyperparamOptManager._get_next(  )

---------------functions---------------


mlmodels\model_tf\raw\temporal_fusion_google\libs\tft_model.py
----------------methods----------------
QuantileLossCalculator.__init__(self, quantiles,   )
QuantileLossCalculator.quantile_loss(self, a, b,   )
TemporalFusionTransformer._batch_single_entity(input_data,   )
TemporalFusionTransformer.convert_real_to_embedding(x,   )
TemporalFusionTransformer.get_lstm(return_state,   )
TemporalFusionTransformer.lstm_combine_and_mask(embedding,   )
TemporalFusionTransformer.static_combine_and_mask(embedding,   )

---------------functions---------------
add_and_norm(x_list,   )
apply_gating_layer(x, hidden_layer_size,  dropout_rate=None, use_time_distributed=True, activation=None,  )
apply_mlp(inputs, hidden_size, output_size,  output_activation=None, hidden_activation='tanh', use_time_distributed=False,  )
gated_residual_network(x, hidden_layer_size,  output_size=None, dropout_rate=None, use_time_distributed=True, additional_context=None, return_gate=False,  )
get_decoder_mask(self_attn_inputs,   )
linear_layer(size,  activation=None, use_time_distributed=False, use_bias=True,  )


mlmodels\model_tf\raw\temporal_fusion_google\libs\utils.py
----------------methods----------------

---------------functions---------------
create_folder_if_not_exist(directory,   )
extract_cols_from_data_type(data_type, column_definition, excluded_input_types,   )
get_default_tensorflow_config( tf_device='gpu', gpu_id=0,  )
get_single_col_by_input_type(input_type, column_definition,   )
load(tf_session, model_folder, cp_name,  scope=None, verbose=False,  )
numpy_normalised_quantile_loss(y, y_pred, quantile,   )
print_weights_in_checkpoint(model_folder, cp_name,   )
save(tf_session, model_folder, cp_name,  scope=None,  )
tensorflow_quantile_loss(y, y_pred, quantile,   )


mlmodels\model_tf\raw\temporal_fusion_google\libs\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\1.turtle-agent.py
----------------methods----------------

---------------functions---------------
buy_stock(real_movement, signal,  initial_money=10000, max_buy=20, max_sell=20,  )


mlmodels\model_tf\raw\tfcode\agent\10.duel-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip, batch_size,   )
Agent.act(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.replay(self, batch_size,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\11.double-duel-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, input_size, output_size, layer_size, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\12.duel-recurrent-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\13.double-duel-recurrent-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, input_size, output_size, layer_size, learning_rate, name,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\14.actor-critic-agent.py
----------------methods----------------
Actor.__init__(self, name, input_size, output_size, size_layer,   )
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories_and_train(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\15.actor-critic-duel-agent.py
----------------methods----------------
Actor.__init__(self, name, input_size, output_size, size_layer,   )
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories_and_train(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\16.actor-critic-recurrent-agent.py
----------------methods----------------
Actor.__init__(self, name, input_size, output_size, size_layer,   )
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories_and_train(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\17.actor-critic-duel-recurrent-agent.py
----------------methods----------------
Actor.__init__(self, name, input_size, output_size, size_layer,   )
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories_and_train(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\18.curiosity-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\19.recurrent-curiosity-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done, rnn_state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\2.moving-average-agent.py
----------------methods----------------

---------------functions---------------
buy_stock(real_movement, signal,  initial_money=10000, max_buy=20, max_sell=20,  )


mlmodels\model_tf\raw\tfcode\agent\20.duel-curiosity-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\21.neuro-evolution-agent.py
----------------methods----------------
NeuroEvolution.__init__(self, population_size, mutation_rate, model_generator, state_size, window_size, trend, skip, initial_money,   )
NeuroEvolution._initialize_population(self,   )
NeuroEvolution.act(self, p, state,   )
NeuroEvolution.buy(self, individual,   )
NeuroEvolution.calculate_fitness(self,   )
NeuroEvolution.crossover(self, parent1, parent2,   )
NeuroEvolution.evolve(self,  generations=20, checkpoint=5,  )
NeuroEvolution.get_state(self, t,   )
NeuroEvolution.inherit_weights(self, parent, child,   )
NeuroEvolution.mutate(self, individual,  scale=1.0,  )
neuralnetwork.__init__(self, id_,  hidden_size=128,  )

---------------functions---------------
feed_forward(X, nets,   )
relu(X,   )
softmax(X,   )


mlmodels\model_tf\raw\tfcode\agent\22.neuro-evolution-novelty-search-agent.py
----------------methods----------------
NeuroEvolution.__init__(self, population_size, mutation_rate, model_generator, state_size, window_size, trend, skip, initial_money,   )
NeuroEvolution._initialize_population(self,   )
NeuroEvolution._memorize(self, q, i, limit,   )
NeuroEvolution.act(self, p, state,   )
NeuroEvolution.buy(self, individual,   )
NeuroEvolution.calculate_fitness(self,   )
NeuroEvolution.crossover(self, parent1, parent2,   )
NeuroEvolution.evaluate(self, individual, backlog, pop,  k=4,  )
NeuroEvolution.evolve(self,  generations=20, checkpoint=5,  )
NeuroEvolution.get_state(self, t,   )
NeuroEvolution.inherit_weights(self, parent, child,   )
NeuroEvolution.mutate(self, individual,  scale=1.0,  )
neuralnetwork.__init__(self, id_,  hidden_size=128,  )

---------------functions---------------
feed_forward(X, nets,   )
relu(X,   )
softmax(X,   )


mlmodels\model_tf\raw\tfcode\agent\3.signal-rolling-agent.py
----------------methods----------------

---------------functions---------------
buy_stock(real_movement,  delay=5, initial_state=1, initial_money=10000, max_buy=20, max_sell=20,  )


mlmodels\model_tf\raw\tfcode\agent\4.policy-gradient-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent.buy(self, initial_money,   )
Agent.discount_rewards(self, r,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\5.q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip, batch_size,   )
Agent.act(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.replay(self, batch_size,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\6.evolution-strategy-agent.py
----------------methods----------------
Agent.__init__(self, model, window_size, trend, skip, initial_money,   )
Agent.act(self, sequence,   )
Agent.buy(self,   )
Agent.fit(self, iterations, checkpoint,   )
Agent.get_reward(self, weights,   )
Agent.get_state(self, t,   )
Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate,   )
Deep_Evolution_Strategy._get_weight_from_population(self, weights, population,   )
Deep_Evolution_Strategy.get_weights(self,   )
Deep_Evolution_Strategy.train(self,  epoch=100, print_every=1,  )
Model.__init__(self, input_size, layer_size, output_size,   )
Model.get_weights(self,   )
Model.predict(self, inputs,   )
Model.set_weights(self, weights,   )

---------------functions---------------
get_imports(  )


mlmodels\model_tf\raw\tfcode\agent\7.double-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, input_size, output_size, layer_size, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\8.recurrent-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\9.double-recurrent-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, input_size, output_size, layer_size, learning_rate, name,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\agent\updated-NES-google.py
----------------methods----------------
Agent.__init__(self, model, money, max_buy, max_sell, close, window_size, skip,   )
Agent.act(self, sequence,   )
Agent.buy(self,   )
Agent.fit(self, iterations, checkpoint,   )
Agent.get_reward(self, weights,   )
Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate,   )
Deep_Evolution_Strategy._get_weight_from_population(self, weights, population,   )
Deep_Evolution_Strategy.get_weights(self,   )
Deep_Evolution_Strategy.train(self,  epoch=100, print_every=1,  )
Model.__init__(self, input_size, layer_size, output_size,   )
Model.get_weights(self,   )
Model.predict(self, inputs,   )
Model.set_weights(self, weights,   )

---------------functions---------------
act(model, sequence,   )
f(w,   )
get_state(data, t, n,   )


mlmodels\model_tf\raw\tfcode\deep-learning\1.lstm.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\10.encoder-vanilla.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )


mlmodels\model_tf\raw\tfcode\deep-learning\11.bidirectional-vanilla.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\12.vanilla-2path.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\13.lstm-seq2seq.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\14.lstm-attention.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\15.lstm-seq2seq-attention.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\16.lstm-seq2seq-bidirectional.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\17.lstm-seq2seq-bidirectional-attention.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\18.lstm-attention-scaleddot.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size, seq_len,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\19.lstm-dilated.py
----------------methods----------------
Model.__init__(self, steps, dimension_input, dimension_output,  learning_rate=0.001, hidden_structs=[20], dilations=[1, 1, 1, 1],  )

---------------functions---------------
anchor(signal, weight,   )
contruct_cells(hidden_structs,   )
dilated_rnn(cell, inputs, rate, states,  scope="default",  )
multi_dilated_rnn(cells, inputs, dilations, states,   )
rnn_reformat(x, input_dims, n_steps,   )


mlmodels\model_tf\raw\tfcode\deep-learning\2.encoder-lstm.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )


mlmodels\model_tf\raw\tfcode\deep-learning\20.only-attention.py
----------------methods----------------
Model.__init__(self, seq_len, learning_rate, dimension_input, dimension_output,   )

---------------functions---------------
anchor(signal, weight,   )
sinusoidal_positional_encoding(inputs, num_units,  zero_pad=False, scale=False,  )


mlmodels\model_tf\raw\tfcode\deep-learning\21.multihead-attention.py
----------------methods----------------
Model.__init__(self, dimension_input, dimension_output, seq_len, learning_rate,  num_heads=8, attn_windows=range(1, 6),  )
Model.multihead_attn(self, inputs, masks,   )
Model.window_mask(self, h_w,   )

---------------functions---------------
anchor(signal, weight,   )
embed_seq(inputs,  vocab_size=None, embed_dim=None, zero_pad=False, scale=False,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_positional_encoding(inputs, embed_dim,  zero_pad=False, scale=False,  )
pointwise_feedforward(inputs,  num_units=[None, None], activation=None,  )


mlmodels\model_tf\raw\tfcode\deep-learning\22.lstm-bahdanau.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\23.lstm-luong.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\24.lstm-luong-bahdanau.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1, attention_size=10,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\25.dnc.py
----------------methods----------------
Model.__init__(self, learning_rate, size, size_layer, output_size,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\deep-learning\26.lstm-residual.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,   )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\27.byte-net.py
----------------methods----------------
ByteNet.__init__(self, size, output_size, channels, encoder_dilations, encoder_filter_width,  learning_rate=0.001, beta1=0.5,  )

---------------functions---------------
anchor(signal, weight,   )
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width,  causal=True,  )
conv1d(input_, output_channels,  dilation=1, filter_width=1, causal=False,  )
layer_normalization(x,  epsilon=1e-8,  )


mlmodels\model_tf\raw\tfcode\deep-learning\28.attention-is-all-you-need.py
----------------methods----------------
Attention.__init__(self, size_layer, embedded_size, learning_rate, size, output_size,  num_blocks=2, num_heads=8, min_freq=50,  )

---------------functions---------------
anchor(signal, weight,   )
label_smoothing(inputs,  epsilon=0.1,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_position_encoding(inputs, mask, embed_dim,   )
multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads,   )
pointwise_feedforward(inputs, hidden_units,  activation=None,  )
sinusoidal_position_encoding(inputs, mask, repr_dim,   )


mlmodels\model_tf\raw\tfcode\deep-learning\29.fairseq.py
----------------methods----------------
Fairseq.__init__(self,   )

---------------functions---------------
anchor(signal, weight,   )
decoder_block(inp, n_hidden, filter_size,   )
encoder_block(inp, n_hidden, filter_size,   )
glu(x,   )
layer(inp, conv_block, kernel_width, n_hidden,  residual=None,  )


mlmodels\model_tf\raw\tfcode\deep-learning\3.bidirectional-lstm.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\4.lstm-2path.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\5.gru.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\6.encoder-gru.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )


mlmodels\model_tf\raw\tfcode\deep-learning\7.bidirectional-gru.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\8.gru-2path.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\9.vanilla.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
anchor(signal, weight,   )


mlmodels\model_tf\raw\tfcode\deep-learning\access.py
----------------methods----------------
MemoryAccess.__init__(self,  memory_size=128, word_size=20, num_reads=1, num_writes=1, name="memory_access",  )
MemoryAccess._build(self, inputs, prev_state,   )
MemoryAccess._read_inputs(self, inputs,   )
MemoryAccess._read_weights(self, inputs, memory, prev_read_weights, link,   )
MemoryAccess._write_weights(self, inputs, memory, usage,   )
MemoryAccess.output_size(self,   )
MemoryAccess.state_size(self,   )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_tf\raw\tfcode\deep-learning\addressing.py
----------------methods----------------
CosineWeights.__init__(self, num_heads, word_size,  strength_op=tf.nn.softplus, name="cosine_weights",  )
CosineWeights._build(self, memory, keys, strengths,   )
Freeness.__init__(self, memory_size,  name="freeness",  )
Freeness._allocation(self, usage,   )
Freeness._build(self, write_weights, free_gate, read_weights, prev_usage,   )
Freeness._usage_after_read(self, prev_usage, free_gate, read_weights,   )
Freeness._usage_after_write(self, prev_usage, write_weights,   )
Freeness.state_size(self,   )
Freeness.write_allocation_weights(self, usage, write_gates, num_writes,   )
TemporalLinkage.__init__(self, memory_size, num_writes,  name="temporal_linkage",  )
TemporalLinkage._build(self, write_weights, prev_state,   )
TemporalLinkage._link(self, prev_link, prev_precedence_weights, write_weights,   )
TemporalLinkage._precedence_weights(self, prev_precedence_weights, write_weights,   )
TemporalLinkage.directional_read_weights(self, link, prev_read_weights, forward,   )
TemporalLinkage.state_size(self,   )

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_tf\raw\tfcode\deep-learning\autoencoder.py
----------------methods----------------

---------------functions---------------
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )


mlmodels\model_tf\raw\tfcode\deep-learning\dnc.py
----------------methods----------------
DNC.__init__(self, access_config, controller_config, output_size,  clip_value=None, name="dnc",  )
DNC._build(self, inputs, prev_state,   )
DNC._clip_if_enabled(self, x,   )
DNC.initial_state(self, batch_size,  dtype=tf.float32,  )
DNC.output_size(self,   )
DNC.state_size(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\deep-learning\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )


mlmodels\model_tf\raw\tfcode\free-agent\evolution-strategy-agent.py
----------------methods----------------
Agent.__init__(self, model, money, max_buy, max_sell,   )
Agent.act(self, sequence,   )
Agent.buy(self,   )
Agent.fit(self, iterations, checkpoint,   )
Agent.get_reward(self, weights,   )
Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate,   )
Deep_Evolution_Strategy._get_weight_from_population(self, weights, population,   )
Deep_Evolution_Strategy.get_weights(self,   )
Deep_Evolution_Strategy.train(self,  epoch=100, print_every=1,  )
Model.__init__(self, input_size, layer_size, output_size,   )
Model.get_weights(self,   )
Model.predict(self, inputs,   )
Model.set_weights(self, weights,   )

---------------functions---------------
get_imports(  )
get_state(data, t, n,   )


mlmodels\model_tf\raw\tfcode\free-agent\evolution-strategy-bayesian-agent.py
----------------methods----------------
Agent.__init__(self, population_size, sigma, learning_rate, model, money, max_buy, max_sell, skip, window_size,   )
Agent.act(self, sequence,   )
Agent.buy(self,   )
Agent.fit(self, iterations, checkpoint,   )
Agent.get_reward(self, weights,   )
Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate,   )
Deep_Evolution_Strategy._get_weight_from_population(self, weights, population,   )
Deep_Evolution_Strategy.get_weights(self,   )
Deep_Evolution_Strategy.train(self,  epoch=100, print_every=1,  )
Model.__init__(self, input_size, layer_size, output_size,   )
Model.get_weights(self,   )
Model.predict(self, inputs,   )
Model.set_weights(self, weights,   )

---------------functions---------------
best_agent(window_size, skip, population_size, sigma, learning_rate, size_network,   )
find_best_agent(window_size, skip, population_size, sigma, learning_rate, size_network,   )
get_imports(  )
get_state(data, t, n,   )


mlmodels\model_tf\raw\tfcode\misc\bitcoin-analysis-lstm.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer,  forget_bias=0.8,  )

---------------functions---------------
anchor(signal, weight,   )
detect(signal,  treshold=2.0,  )
df_shift(df,  lag=0, start=1, skip=1, rejected_columns=[],  )
moving_average(signal, period,   )
predict_future(future_count, df, dates,  indices={},  )


mlmodels\model_tf\raw\tfcode\misc\fashion-forecasting.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer,  forget_bias=0.8,  )

---------------functions---------------
df_shift(df,  lag=0, rejected_columns=[],  )


mlmodels\model_tf\raw\tfcode\misc\kijang-emas-bank-negara.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )
Model.change_percentage(val,   )

---------------functions---------------
calculate_accuracy(real, predict,   )
calculate_distance(real, predict,   )
detect(signal,  treshold=2.0,  )
df_shift(df,  lag=0, start=1, skip=1, rejected_columns=[],  )


mlmodels\model_tf\raw\tfcode\misc\outliers.py
----------------methods----------------

---------------functions---------------
df_shift(df,  lag=0, start=1, skip=1, rejected_columns=[],  )
getDistanceByPoint(data, model,   )


mlmodels\model_tf\raw\tfcode\misc\tesla-study.py
----------------methods----------------

---------------functions---------------
adf(ts,   )


mlmodels\model_tf\raw\tfcode\simulation\mcmc-stock-market.py
----------------methods----------------

---------------functions---------------
acceptance(x, x_new,   )
log_gamma(x, data,   )
log_norm(x, data,   )
log_skewnorm(x, data,   )
metropolis_hastings(pdf, trans_model, param_init, iterations, data,   )
moving_average(signal, period,   )
pct_change(x,  period=1,  )
prior(x,   )


mlmodels\model_tf\raw\tfcode\simulation\stock-forecasting-monte-carlo.py
----------------methods----------------

---------------functions---------------
pct_change(x,  period=1,  )


mlmodels\model_tf\raw\tfcode\stacking\autoencoder.py
----------------methods----------------

---------------functions---------------
reducedimension(input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )


mlmodels\model_tf\raw\tfcode\stacking\model.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------


mlmodels\model_tf\raw\tfcode\stacking\stack-encoder-ensemble-xgb.py
----------------methods----------------
encoder.__init__(self, input_,  dimension=2, learning_rate=0.01, hidden_layer=256, epoch=20,  )
encoder.encode(self, input_,   )

---------------functions---------------
predict(count,  history=5,  )
reverse_close(array,   )


mlmodels\model_tf\raw\tfcode\stacking\stack-rnn-arima-xgb.py
----------------methods----------------

---------------functions---------------
reverse_close(array,   )


mlmodels\model_tf\raw\tfcode2\Attention\1.bahdanau.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Bahdanau.__call__(self, inputs, state,  scope=None,  )
Bahdanau.__init__(self, hidden_size, output_size, encoder_outputs,   )
Bahdanau.get_attention(self, inputs, state,   )
Bahdanau.output_size(self,   )
Bahdanau.reset_state(self,   )
Bahdanau.state_size(self,   )
Model.__init__(self, size_layer, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Attention\2.luong.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Luong.__call__(self, inputs, state,  scope=None,  )
Luong.__init__(self, hidden_size, output_size, encoder_outputs,   )
Luong.output_size(self,   )
Luong.reset_state(self,   )
Luong.state_size(self,   )
Model.__init__(self, size_layer, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Attention\3.hierarchical.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Attention\4.additive.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Attention\5.soft.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Attention\6.attention-over-attention.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Attention\7.bahdanau-api.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Attention\8.luong-api.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Bayesian-Optimization\bayesian-cnn-cifar10.py
----------------methods----------------

---------------functions---------------
generate_nn(fully_conn_size, len_layer_conv, kernel_size, learning_rate, pooling_size, multiply, dropout_rate, beta, activation, batch_normalization,   )
neural_network(fully_conn_size, len_layer_conv, kernel_size, learning_rate, pooling_size, multiply, dropout_rate, beta, activation, batch_normalization,  batch_size=10,  )
reshape_image(img,   )
unpickle(file,   )


mlmodels\model_tf\raw\tfcode2\Bayesian-Optimization\bayesian-feedforward-iris.py
----------------methods----------------

---------------functions---------------
generate_nn(num_hidden, size_layer, learning_rate, dropout_rate, beta, activation,   )
neural_network(num_hidden, size_layer, learning_rate, dropout_rate, beta, activation,  batch_size=16,  )


mlmodels\model_tf\raw\tfcode2\Bayesian-Optimization\bayesian-recurrent-sentiment.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
generate_nn(num_hidden, size_layer, learning_rate, dropout_rate, beta, activation, seq_len,   )
neural_network(num_hidden, size_layer, learning_rate, dropout_rate, beta, activation, seq_len,  batch_size=20,  )


mlmodels\model_tf\raw\tfcode2\Bayesian-Optimization\cnn_iceberg.py
----------------methods----------------

---------------functions---------------
generate_nn(fully_conn_size, len_layer_conv, kernel_size, learning_rate, pooling_size, multiply, dropout_rate, beta, activation, batch_normalization,   )
neural_network(fully_conn_size,   )


mlmodels\model_tf\raw\tfcode2\CNN\alex-net\alexnet.py
----------------methods----------------
Alexnet.__init__(self, input_size, output_dimension, learning_rate,   )

---------------functions---------------
unpickle(file,   )


mlmodels\model_tf\raw\tfcode2\CNN\binary-net\binary-net.py
----------------methods----------------
Model.__init__(self,   )

---------------functions---------------
activation(x,   )
batch_norm(x, epsilon,  decay=0.9, is_training=True,  )
layer(x, filter_output,  filter_size=[1, 1], stride=[1, 1], pool=None, activate="bin", norm=True, epsilon=0.0001, padding="SAME",  )
weight_bias(shape,   )


mlmodels\model_tf\raw\tfcode2\CNN\bytenet-translator\bytenet-translator.py
----------------methods----------------
ByteNet.__init__(self, from_vocab_size, to_vocab_size, channels, encoder_dilations, decoder_dilations, encoder_filter_width, decoder_filter_width,  learning_rate=0.001, beta1=0.5,  )

---------------functions---------------
build_dataset(words, n_words,   )
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width,  causal=True,  )
conv1d(input_, output_channels,  dilation=1, filter_width=1, causal=False,  )
create_buckets(text_from, text_to, bucket_quant, from_vocab, to_vocab,   )
get_batch_from_pairs(pair_list,   )
layer_normalization(x,  epsilon=1e-8,  )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tfcode2\CNN\capsule-network\capsule-mnist.py
----------------methods----------------
CapsuleNetwork.__init__(self, batch_size, learning_rate,  regularization_scale=0.392, epsilon=1e-8, m_plus=0.9, m_minus=0.1, lambda_val=0.5,  )

---------------functions---------------
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output,   )
routing(X, b_IJ,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )


mlmodels\model_tf\raw\tfcode2\CNN\densenet\dense-net.py
----------------methods----------------
DenseNet.Dense_net(self, input_x,   )
DenseNet.__init__(self, x, nb_blocks, filters, training,   )
DenseNet.bottleneck_layer(self, x, scope,   )
DenseNet.dense_block(self, input_x, nb_layers, layer_name,   )
DenseNet.transition_layer(self, x, scope,   )

---------------functions---------------
average_pooling(x,  pool_size=[2, 2], stride=2, padding="VALID",  )
batch_normalization(x, training, scope,   )
concatenation(layers,   )
conv_layer(input, filter, kernel,  stride=1, layer_name="conv",  )
drop_out(x, rate, training,   )
global_average_pooling(x,  stride=1,  )
linear(x,   )
max_pooling(x,  pool_size=[3, 3], stride=2, padding="VALID",  )


mlmodels\model_tf\raw\tfcode2\CNN\humming-network\ghn-mnist.py
----------------methods----------------
Model.__init__(self, learning_rate,   )

---------------functions---------------
conv(inputs, filters, kernel_size,   )
differentiable_clip(inputs, alpha, rmin, rmax,   )
double_thresholding(inputs,  per_pixel=True,  )
fully_connected(inputs, out_size,   )


mlmodels\model_tf\raw\tfcode2\CNN\kmax-conv1d\kmax-conv-mnist.py
----------------methods----------------
Model.__init__(self,  learning_rate=1e-4, top_k=5, n_filters=250,  )
Model.add_kmax_pooling(self, x,   )

---------------functions---------------
add_conv1d(x, n_filters, kernel_size,  strides=1,  )


mlmodels\model_tf\raw\tfcode2\CNN\kmax-conv1d\kmax-conv.py
----------------methods----------------
Model.__init__(self, seq_len, dimension_input, dimension_output, learning_rate,  top_k=5, n_filters=250,  )
Model.add_kmax_pooling(self, x,   )

---------------functions---------------
add_conv1d(x, n_filters, kernel_size,  strides=1,  )


mlmodels\model_tf\raw\tfcode2\CNN\siamese-network\siamese-network.py
----------------methods----------------
Siamese.__init__(self,   )

---------------functions---------------
convolutionize(x, conv_w,  h=1,  )
create_network(X,  scope="conv", reuse=False,  )
pooling(wx,   )


mlmodels\model_tf\raw\tfcode2\CNN\temporal-conv1d\temporal-conv.py
----------------methods----------------
Model.__init__(self,  filters=32, kernel_size=4, dilations=[1, 2, 4, 8], stacks=8,  )

---------------functions---------------
residual_block(x, i, filters, kernel_size,   )


mlmodels\model_tf\raw\tfcode2\CNN\triplet-loss\batch_all_triplet_loss.py
----------------methods----------------
Siamese.__init__(self,   )

---------------functions---------------
_get_anchor_negative_triplet_mask(labels,   )
_get_anchor_positive_triplet_mask(labels,   )
_get_triplet_mask(labels,   )
_pairwise_distances(embeddings,  squared=False,  )
batch_all_triplet_loss(labels, embeddings, margin,  squared=False,  )
convolutionize(x, conv_w,  h=1,  )
create_network(X,  scope="conv", reuse=False,  )
pooling(wx,   )


mlmodels\model_tf\raw\tfcode2\CNN\triplet-loss\batch_hard_triplet_loss.py
----------------methods----------------
Siamese.__init__(self,   )

---------------functions---------------
_get_anchor_negative_triplet_mask(labels,   )
_get_anchor_positive_triplet_mask(labels,   )
_get_triplet_mask(labels,   )
_pairwise_distances(embeddings,  squared=False,  )
batch_hard_triplet_loss(labels, embeddings, margin,  squared=False,  )
convolutionize(x, conv_w,  h=1,  )
create_network(X,  scope="conv", reuse=False,  )
pooling(wx,   )


mlmodels\model_tf\raw\tfcode2\CNN\triplet-loss\triplet-loss.py
----------------methods----------------
Siamese.__init__(self,   )

---------------functions---------------
compute_euclidean_distance(x, y,   )
compute_triplet_loss(anchor_feature, positive_feature, negative_feature,  margin=0.01,  )
convolutionize(x, conv_w,  h=1,  )
create_network(X,  scope="conv", reuse=False,  )
get_one_triplet(input_data, input_labels, n_labels,   )
pooling(wx,   )


mlmodels\model_tf\raw\tfcode2\CNN\unet\unet.py
----------------methods----------------
Model.__init__(self,   )

---------------functions---------------
IOU_calc(y_true, y_pred,   )
IOU_calc_loss(y_true, y_pred,   )
conv_layer(x, conv, out_shape, name,  stride=1,  )
plot_image(img, img_mask, bboxes,   )
pooling(x,  k=2, stride=2,  )
read_img(img_name,   )


mlmodels\model_tf\raw\tfcode2\distributed\1.tf-distributed\tf-distributed.py
----------------methods----------------

---------------functions---------------
main(_,   )
variable_summaries(var,   )


mlmodels\model_tf\raw\tfcode2\distributed\2.pyspark-sparkflow\1.mnist-sparkflow-feedforward.py
----------------methods----------------

---------------functions---------------
download_from_url(url, dst,   )
small_model(  )


mlmodels\model_tf\raw\tfcode2\distributed\2.pyspark-sparkflow\2.mnist-sparkflow-cnn.py
----------------methods----------------

---------------functions---------------
cnn_model(  )
download_from_url(url, dst,   )


mlmodels\model_tf\raw\tfcode2\distributed\3.dask-tensorflow\dask-tensorflow.py
----------------methods----------------

---------------functions---------------
get_mnist(  )
model(server,   )
ps_task(  )
scoring_task(  )
transfer_dask_to_tensorflow(batch,   )
worker_task(  )


mlmodels\model_tf\raw\tfcode2\Feed-Forward\addsign-powersign\AddSign.py
----------------methods----------------
AddSign.__init__(self,  learning_rate=1.001, alpha=0.01, beta=0.5, use_locking=False, name="AddSign",  )
AddSign._apply_dense(self, grad, var,   )
AddSign._apply_sparse(self, grad, var,   )
AddSign._create_slots(self, var_list,   )
AddSign._prepare(self,   )
Model.__init__(self,  learning_rate=0.01,  )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Feed-Forward\addsign-powersign\PowerSign.py
----------------methods----------------
Model.__init__(self,  learning_rate=0.01,  )
PowerSign.__init__(self,  learning_rate=1.001, alpha=0.01, beta=0.5, use_locking=False, name="AddSign",  )
PowerSign._apply_dense(self, grad, var,   )
PowerSign._apply_sparse(self, grad, var,   )
PowerSign._create_slots(self, var_list,   )
PowerSign._prepare(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Feed-Forward\batch-normalization\batch-normalization.py
----------------methods----------------
NeuralNet.__init__(self, initial_weights, activation_fn, use_batch_norm,   )
NeuralNet.build_network(self, initial_weights, activation_fn,   )
NeuralNet.fully_connected(self, layer_in, initial_weights,  activation_fn=None,  )
NeuralNet.test(self, session,  test_training_accuracy=False, include_individual_predictions=False, restore_from=None,  )
NeuralNet.train(self, session, learning_rate, training_batches, batches_per_sample,  save_model_as=None,  )

---------------functions---------------
plot_training_accuracies(  *args, **kwargs)
train_and_test(use_bad_weights, learning_rate, activation_fn,  training_batches=50000, batches_per_sample=500,  )


mlmodels\model_tf\raw\tfcode2\Feed-Forward\cocob\cocob.py
----------------methods----------------
COCOB.__init__(self,  alpha=100, use_locking=False, name="COCOB",  )
COCOB._apply_dense(self, grad, var,   )
COCOB._apply_sparse(self, grad, var,   )
COCOB._create_slots(self, var_list,   )
COCOB._resource_apply_dense(self, grad, handle,   )
Model.__init__(self,  learning_rate=0.01,  )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Feed-Forward\dropout-comparison\dropout-comparison-Iris.py
----------------methods----------------
Dropout_model.__init__(self, learning_rate, layer_size,   )
Normal_model.__init__(self, learning_rate, layer_size,   )

---------------functions---------------
training(epoch,   )


mlmodels\model_tf\raw\tfcode2\Feed-Forward\gradient-comparison\Gradientcomparison-Iris.py
----------------methods----------------
Model.__init__(self, learning_rate, layer_size, optimizer,   )

---------------functions---------------
training(epoch,   )


mlmodels\model_tf\raw\tfcode2\Feed-Forward\regularization-comparison\regularization-comparison-Iris.py
----------------methods----------------
L1L2_model.__init__(self, learning_rate, layer_size, alpha,   )
L1_model.__init__(self, learning_rate, layer_size, alpha,   )
L2_model.__init__(self, learning_rate, layer_size, alpha,   )
Normal_model.__init__(self, learning_rate, layer_size,   )

---------------functions---------------
training(epoch,   )


mlmodels\model_tf\raw\tfcode2\Feed-Forward\self-normalized\mnist-withAPI.py
----------------methods----------------
Model.__init__(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Feed-Forward\self-normalized\mnist-withoutAPI.py
----------------methods----------------
Model.__init__(self,   )

---------------functions---------------
alpha_dropout(x, rate,  alpha=-1.7580993408473766, mean=0.0, var=1.0,  )
selu(x,   )


mlmodels\model_tf\raw\tfcode2\Feed-Forward\word-vector-skipgram\skipgram-nce.py
----------------methods----------------
DLProgress.hook(self,  block_num=1, block_size=1, total_size=None,  )
Model.__init__(self, batch_size, dimension_size, learning_rate, vocabulary_size,   )

---------------functions---------------
build_dataset(words, vocabulary_size,   )
generate_batch_skipgram(words, batch_size, num_skips, skip_window,   )


mlmodels\model_tf\raw\tfcode2\GAN\DCGAN\DCGAN.py
----------------methods----------------
DLProgress.hook(self,  block_num=1, block_size=1, total_size=None,  )
Model.__init__(self, z_dim,  learning_rate=0.0001, beta=0.5, alpha=0.2,  )

---------------functions---------------
first_originate(x,   )
first_scale(x,  feature_range=(-1, 1),  )
generate_sample(samples,   )
second_originate(x,   )
second_scale(x,   )


mlmodels\model_tf\raw\tfcode2\GAN\DCGAN\DCGAN-model\first_model.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\GAN\DCGAN\DCGAN-model\fourth_model.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\GAN\DCGAN\DCGAN-model\model.py
----------------methods----------------
Model.__init__(self, learning_rate, batch_size, dimension_picture, color_layer,  output_dimension=64,  )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\GAN\DCGAN\DCGAN-model\second_model.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\GAN\DCGAN\DCGAN-model\third_model.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\GAN\discoGAN-Fashion-mnist\discogan.py
----------------methods----------------
discoGAN.__init__(self,  learning_rate=1e-6,  )

---------------functions---------------
discriminator(z, name,  reuse=False,  )
generate_sample(samples,   )
generator(z, name,  reuse=False, training=True,  )
huber_loss(logits, labels,  max_gradient=1.0,  )
train(model, X, Y, batch, epoch,   )


mlmodels\model_tf\raw\tfcode2\hybrid\cnn-rnn-lstm\cnn-lstm-ctc.py
----------------methods----------------
Model.__init__(self,   )
Model.batch_norm(self, name, x,   )
Model.conv2d(self, x, name, filter_size, channel_in, channel_out, strides,   )
Model.leaky_relu(self, x,  leak=0,  )
Model.max_pool(self, x, size, strides,   )

---------------functions---------------
accuracy_calculation(original_seq, decoded_seq,  ignore_value=-1,  )
sparse_tuple_from_label(sequences,  dtype=np.int32,  )


mlmodels\model_tf\raw\tfcode2\hybrid\GAN-Sentence\main.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\hybrid\GAN-Sentence\model.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, len_noise, sequence_size, learning_rate,   )

---------------functions---------------
discriminator(X, num_layers, size_layer, dimension_input,  reuse=False,  )
generate(sess, sequence, noise, model, tag, length_sentence, text_vocab,   )
generator_encode(X, num_layers, size_layer, len_noise,  reuse=False,  )
generator_sentence(X, hidden_layer, num_layers, size_layer, dimension_input, name_scope,  reuse=False,  )
lstm_cell(size_layer,  state_is_tuple=True,  )


mlmodels\model_tf\raw\tfcode2\hybrid\GAN-Sentence\parse.py
----------------methods----------------

---------------functions---------------
embed_to_onehot(data, vocab,   )
get_vocab(data_location,   )


mlmodels\model_tf\raw\tfcode2\Misc\1.20newsgroup.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tfcode2\Misc\4.tfrecords.py
----------------methods----------------

---------------functions---------------
_bytes_feature(value,   )
_int64_feature(value,   )
convert_to(dataset, labels, name, i,   )
convolutionize(x, conv_w,  h=1,  )
create_network(X,  scope="conv", reuse=False,  )
pooling(wx,   )
read_and_decode(filename_queue,   )


mlmodels\model_tf\raw\tfcode2\Misc\2.debugger\mnist-debugger.py
----------------methods----------------

---------------functions---------------
convolutionize(x, conv_w,  h=1,  )
feed_dict(train,   )
pooling(wx,   )


mlmodels\model_tf\raw\tfcode2\Misc\3.emotion-mobilenet\conv_blocks.py
----------------methods----------------

---------------functions---------------
_fixed_padding(inputs, kernel_size,  rate=1,  )
_make_divisible(v, divisor,  min_value=None,  )
_split_divisible(num, num_ways,  divisible_by=8,  )
_v1_compatible_scope_naming(scope,   )
expand_input_by_factor(n,  divisible_by=8,  )
expanded_conv(input_tensor, num_outputs,  expansion_size=expand_input_by_factor(6), stride=1, rate=1, kernel_size=(3, 3), residual=True, normalizer_fn=None, split_projection=1, split_expansion=1, expansion_transform=None, depthwise_location="expansion", depthwise_channel_multiplier=1, endpoints=None, use_explicit_padding=False, padding="SAME", scope=None,  )
split_conv(input_tensor, num_outputs, num_ways, scope,  divisible_by=8,  **kwargs)
split_separable_conv2d(input_tensor, num_outputs,  scope=None, normalizer_fn=None, stride=1, rate=1, endpoints=None, use_explicit_padding=False,  )


mlmodels\model_tf\raw\tfcode2\Misc\3.emotion-mobilenet\mobilenet.py
----------------methods----------------
NoOpScope.__enter__(self,   )
NoOpScope.__exit__(self, exc_type, exc_value, traceback,   )

---------------functions---------------
_fixed_padding(inputs, kernel_size,  rate=1,  )
_make_divisible(v, divisor,  min_value=None,  )
_scope_all(scope,  default_scope=None,  )
_set_arg_scope_defaults(defaults,   )
apply_activation(x,  name=None, activation_fn=None,  )
depth_multiplier(output_params, multiplier,  divisible_by=8, min_depth=8,  **unused_kwargs)
global_pool(input_tensor,  pool_op=tf.nn.avg_pool,  )
mobilenet(inputs,  num_classes=1001, prediction_fn=slim.softmax, reuse=None, scope="Mobilenet", base_only=False,  **mobilenet_args)
mobilenet_base(  )
op(opfunc,   **params)
safe_arg_scope(funcs,   **kwargs)
training_scope( is_training=True, weight_decay=0.00004, stddev=0.09, dropout_keep_prob=0.8, bn_decay=0.997,  )


mlmodels\model_tf\raw\tfcode2\Misc\3.emotion-mobilenet\mobilenet_v2.py
----------------methods----------------

---------------functions---------------
mobilenet(input_tensor,  num_classes=1001, depth_multiplier=1.0, scope="MobilenetV2", conv_defs=None, finegrain_classification_mode=False, min_depth=None, divisible_by=None,  **kwargs)
mobilenet_base(input_tensor,  depth_multiplier=1.0,  **kwargs)
training_scope(  **kwargs)


mlmodels\model_tf\raw\tfcode2\Misc\3.emotion-mobilenet\transfer-learning.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\save-serving.py
----------------methods----------------
Model.__init__(self, learning_rate, y_shape,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\classification_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\get_model_metadata_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\inference_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\input_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\model_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\model_service_pb2.py
----------------methods----------------
BetaModelServiceServicer.GetModelStatus(self, request, context,   )
BetaModelServiceStub.GetModelStatus(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
ModelServiceServicer.GetModelStatus(self, request, context,   )
ModelServiceStub.__init__(self, channel,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\model_service_pb2_grpc.py
----------------methods----------------
ModelServiceServicer.GetModelStatus(self, request, context,   )
ModelServiceStub.__init__(self, channel,   )

---------------functions---------------
add_ModelServiceServicer_to_server(servicer, server,   )


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\prediction_service_pb2.py
----------------methods----------------
BetaPredictionServiceServicer.Classify(self, request, context,   )
BetaPredictionServiceServicer.GetModelMetadata(self, request, context,   )
BetaPredictionServiceServicer.MultiInference(self, request, context,   )
BetaPredictionServiceServicer.Predict(self, request, context,   )
BetaPredictionServiceServicer.Regress(self, request, context,   )
BetaPredictionServiceStub.Classify(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
BetaPredictionServiceStub.GetModelMetadata(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
BetaPredictionServiceStub.MultiInference(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
BetaPredictionServiceStub.Predict(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
BetaPredictionServiceStub.Regress(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
PredictionServiceServicer.Classify(self, request, context,   )
PredictionServiceServicer.GetModelMetadata(self, request, context,   )
PredictionServiceServicer.MultiInference(self, request, context,   )
PredictionServiceServicer.Predict(self, request, context,   )
PredictionServiceServicer.Regress(self, request, context,   )
PredictionServiceStub.__init__(self, channel,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\predict_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\regression_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\5.tf-serving\tensorflow_serving\apis\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\6.renaming-checkpoint\shadow-net-reverse-checkpoint.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\6.renaming-checkpoint\shadow-net.py
----------------methods----------------
Model.__init__(self,   )

---------------functions---------------
sparse_tensor_to_str(sparse_tensor,   )
sparse_tensor_to_str2(spares_tensor,   )


mlmodels\model_tf\raw\tfcode2\Misc\7.load-inception\inception-v1.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\7.load-inception\inception-v3.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Misc\8.inception-visualization\inception-v1-learning.py
----------------methods----------------

---------------functions---------------
deprocess_image(x,   )
generate_pattern(layer_name, filter_index,  size=150, epoch=40,  )


mlmodels\model_tf\raw\tfcode2\Regression\elasticnet-regression-tensorflow.py
----------------methods----------------
Elasticnet.__init__(self, learning_rate, alpha,   )

---------------functions---------------
gradient_mean_square(epoch,   )


mlmodels\model_tf\raw\tfcode2\Regression\lasso-regression-tensorflow.py
----------------methods----------------
Lasso.__init__(self, learning_rate, alpha,   )

---------------functions---------------
gradient_mean_square(epoch,   )


mlmodels\model_tf\raw\tfcode2\Regression\linear-regression-tensorflow.py
----------------methods----------------
Linear.__init__(self, learning_rate,   )

---------------functions---------------
gradient_mean_square(epoch,   )


mlmodels\model_tf\raw\tfcode2\Regression\polynomial-regression-tensorflow.py
----------------methods----------------
Polynomial.__init__(self, learning_rate,   )

---------------functions---------------
gradient_mean_square(epoch,   )


mlmodels\model_tf\raw\tfcode2\Regression\quantile-regression.py
----------------methods----------------
q_model.__init__(self, sess, quantiles,  in_shape=1, out_shape=1, batch_size=32,  )
q_model.build_model(self,  scope="q_model", reuse=tf.AUTO_REUSE,  )
q_model.fit(self, x, y,  epochs=100,  )
q_model.predict(self, x,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\Regression\ridge-regression-tensorflow.py
----------------methods----------------
Ridge.__init__(self, learning_rate, alpha,   )

---------------functions---------------
gradient_mean_square(epoch,   )


mlmodels\model_tf\raw\tfcode2\Regression\sigmoid-regression-tensorflow.py
----------------methods----------------
Logistic.__init__(self, learning_rate,   )

---------------functions---------------
gradient_mean_square(epoch,   )


mlmodels\model_tf\raw\tfcode2\RNN\Dilated-RNN\dilated-rnn-lstm.py
----------------methods----------------
Model.__init__(self, steps, dimension_input, dimension_output,  learning_rate=0.0001, hidden_structs=[20], dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],  )

---------------functions---------------
contruct_cells(hidden_structs,   )
dilated_rnn(cell, inputs, rate,  scope="default",  )
multi_dilated_rnn(cells, inputs, dilations,   )
rnn_reformat(x, input_dims, n_steps,   )


mlmodels\model_tf\raw\tfcode2\RNN\DNC\access.py
----------------methods----------------
MemoryAccess.__init__(self,  memory_size=128, word_size=20, num_reads=1, num_writes=1, name="memory_access",  )
MemoryAccess._build(self, inputs, prev_state,   )
MemoryAccess._read_inputs(self, inputs,   )
MemoryAccess._read_weights(self, inputs, memory, prev_read_weights, link,   )
MemoryAccess._write_weights(self, inputs, memory, usage,   )
MemoryAccess.output_size(self,   )
MemoryAccess.state_size(self,   )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_tf\raw\tfcode2\RNN\DNC\addressing.py
----------------methods----------------
CosineWeights.__init__(self, num_heads, word_size,  strength_op=tf.nn.softplus, name="cosine_weights",  )
CosineWeights._build(self, memory, keys, strengths,   )
Freeness.__init__(self, memory_size,  name="freeness",  )
Freeness._allocation(self, usage,   )
Freeness._build(self, write_weights, free_gate, read_weights, prev_usage,   )
Freeness._usage_after_read(self, prev_usage, free_gate, read_weights,   )
Freeness._usage_after_write(self, prev_usage, write_weights,   )
Freeness.state_size(self,   )
Freeness.write_allocation_weights(self, usage, write_gates, num_writes,   )
TemporalLinkage.__init__(self, memory_size, num_writes,  name="temporal_linkage",  )
TemporalLinkage._build(self, write_weights, prev_state,   )
TemporalLinkage._link(self, prev_link, prev_precedence_weights, write_weights,   )
TemporalLinkage._precedence_weights(self, prev_precedence_weights, write_weights,   )
TemporalLinkage.directional_read_weights(self, link, prev_read_weights, forward,   )
TemporalLinkage.state_size(self,   )

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_tf\raw\tfcode2\RNN\DNC\dnc.py
----------------methods----------------
Model.__init__(self, dict_size,   )

---------------functions---------------
run_model(input_sequence, output_size,   )


mlmodels\model_tf\raw\tfcode2\RNN\DNC\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )


mlmodels\model_tf\raw\tfcode2\RNN\Fast-slow-LSTM\fast-slow-lstm.py
----------------methods----------------
FSRNNCell.__call__(self, inputs, state,  scope="FS-RNN",  )
FSRNNCell.__init__(self, fast_cells, slow_cell,  keep_prob=1.0, training=True,  )
FSRNNCell.zero_state(self, batch_size, dtype,   )
LN_LSTMCell.__call__(self, x, state,  scope=None,  )
LN_LSTMCell.__init__(self, num_units,  f_bias=1.0, use_zoneout=False, zoneout_keep_h=0.9, zoneout_keep_c=0.5, is_training=False,  )
LN_LSTMCell.zero_state(self, batch_size, dtype,   )
Model.__init__(self, size_layer, num_layers, fast_layers, embedded_size, dict_size, dimension_output, learning_rate, batch_size, timestamp,  is_training=True, zoneout_h=0.95, zoneout_c=0.7, keep_prob=0.75,  )

---------------functions---------------
layer_norm(x,  scope="layer_norm", alpha_start=1.0, bias_start=0.0,  )
layer_norm_all(h, base, num_units, scope,   )
moments_for_layer_norm(x,  axes=1, name=None,  )
zoneout(new_h, new_c, h, c, h_keep, c_keep, is_training,   )


mlmodels\model_tf\raw\tfcode2\RNN\Generator-Comparison\generator-comparison.py
----------------methods----------------
Model_GRU.__init__(self, num_layers, size_layer, dimension, learning_rate,   )
Model_GRU_Bi.__init__(self, num_layers, size_layer, dimension, learning_rate,   )
Model_LSTM.__init__(self, num_layers, size_layer, dimension, learning_rate,   )
Model_LSTM_Bi.__init__(self, num_layers, size_layer, dimension, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_sequence(sess, model, tag,  gru=False,  )
generate_based_sequence_bi(sess, model, tag,  gru=False,  )
get_vocab(file,  lower=False,  )
train_random_sequence(sess, model, init_value,   )
train_random_sequence_bi(sess, model, init_value_fw, init_value_bw,   )
training(epoch,   )


mlmodels\model_tf\raw\tfcode2\RNN\LN-LSTM\LNLSTM.py
----------------methods----------------
LN_LSTMCell.__init__(self, num_units,  f_bias=1.0, use_zoneout=False, zoneout_keep_h=0.9, zoneout_keep_c=0.5, is_training=True, reuse=None, name=None,  )
LN_LSTMCell.build(self, inputs_shape,   )
LN_LSTMCell.call(self, x, state,   )
LN_LSTMCell.output_size(self,   )
LN_LSTMCell.state_size(self,   )
LN_LSTMCell.zero_state(self, batch_size, dtype,   )
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------
layer_norm(x,  scope="layer_norm", alpha_start=1.0, bias_start=0.0,  )
layer_norm_all(h, base, num_units, scope,   )
moments_for_layer_norm(x,  axes=1, name=None,  )
zoneout(new_h, new_c, h, c, h_keep, c_keep, is_training,   )


mlmodels\model_tf\raw\tfcode2\RNN\Multihead-Attention\multihead-attention-mnist.py
----------------methods----------------
Model.__init__(self,  dimension_output=10, learning_rate=1e-4, seq_len=28, dimension_input=28, num_heads=28, attn_windows=range(1, 6),  )
Model.multihead_attn(self, inputs, masks,   )
Model.window_mask(self, h_w,   )

---------------functions---------------
embed_seq(inputs,  vocab_size=None, embed_dim=None, zero_pad=False, scale=False,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_positional_encoding(inputs, embed_dim,  zero_pad=False, scale=False,  )
pointwise_feedforward(inputs,  num_units=[None, None], activation=None,  )


mlmodels\model_tf\raw\tfcode2\RNN\Music-Generator\generate-music.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
create_midi(prediction_output, output_name,   )
embed_to_onehot(data, vocab,   )
generate_based_sequence(length_sentence,  argmax=False, temp=1,  )
get_notes(  )
train_random_sequence(  )


mlmodels\model_tf\raw\tfcode2\RNN\Nested-LSTM\nested-lstm.py
----------------methods----------------
Model.__init__(self, size_layer, embedded_size, dict_size, dimension_output, learning_rate, batch_size, timestamp,  depth=1,  )
NLSTMCell.__init__(self, num_units, depth,  forget_bias=1.0, state_is_tuple=True, use_peepholes=True, activation=None, gate_activation=None, cell_activation=None, initializer=None, input_gate_initializer=None, use_bias=True, reuse=None, name=None,  )
NLSTMCell._recurrence(self, inputs, hidden_state, cell_states, depth,   )
NLSTMCell.build(self, inputs_shape,   )
NLSTMCell.call(self, inputs, state,   )
NLSTMCell.depth(self,   )
NLSTMCell.output_size(self,   )
NLSTMCell.state_size(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\RNN\Neural-Turing-Machine\text-classification.py
----------------methods----------------
Model.__init__(self, seq_len, size_layer, batch_size, dimension_input, dimension_output, learning_rate, memory_size, memory_vector_size,  read_head_num=4, write_head_num=1,  )
NTMCell.__call__(self, x, prev_state,   )
NTMCell.__init__(self, rnn_size, memory_size, memory_vector_dim, read_head_num, write_head_num,  addressing_mode="content_and_location", shift_range=1, reuse=False, output_dim=None,  )
NTMCell.addressing(self, k, beta, g, s, gamma, prev_M, prev_w,   )
NTMCell.zero_state(self, batch_size, dtype,   )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\RNN\Only-Attention\only-attention-mnist.py
----------------methods----------------
Model.__init__(self,   )

---------------functions---------------
sinusoidal_positional_encoding(inputs, num_units,  zero_pad=False, scale=False,  )


mlmodels\model_tf\raw\tfcode2\RNN\Only-Attention\only-attention.py
----------------methods----------------
Model.__init__(self, seq_len, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------
sinusoidal_positional_encoding(inputs, num_units,  zero_pad=False, scale=False,  )


mlmodels\model_tf\raw\tfcode2\RNN\Siamese-network\siamese-network.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output,  margin=0.2,  )

---------------functions---------------


mlmodels\model_tf\raw\tfcode2\RNN\SRU\simple-recurrent-units.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output,   )
SRUCell.__call__(self, inputs, state,  scope="SRUCell",  )
SRUCell.__init__(self, num_units,  activation=None, reuse=None,  )
SRUCell.output_size(self,   )
SRUCell.state_size(self,   )

---------------functions---------------
linear(args, output_size, bias,  bias_start=0.0, scope=None,  )


mlmodels\model_tf\raw\tfcode2\RNN\Stock-forecasting\stock-forecasting.py
----------------methods----------------
Model.__init__(self, learning_rate, num_layers, size, size_layer, output_size,  forget_bias=0.1,  )

---------------functions---------------
train(epoch,   )


mlmodels\model_tf\raw\tfcode2\RNN\T-LSTM\t-lstm.py
----------------methods----------------
TLSTM.TLSTM_Unit(self, prev_hidden_memory, concat_input,   )
TLSTM.__init__(self, input_dim, output_dim, hidden_dim, fc_dim,   )
TLSTM.get_cost_acc(self,   )
TLSTM.get_output(self, state,   )
TLSTM.get_outputs(self,   )
TLSTM.get_states(self,   )
TLSTM.init_bias(self, output_dim, name,   )
TLSTM.init_weights(self, input_dim, output_dim, name,  std=0.1, reg=None,  )
TLSTM.map_elapse_time(self, t,   )
TLSTM.no_init_bias(self, output_dim, name,   )
TLSTM.no_init_weights(self, input_dim, output_dim, name,   )

---------------functions---------------
load_pkl(path,   )


mlmodels\model_tf\raw\tfcode2\RNN\Text-Generator\generate-kernel-lstm.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_length(length_sentence,  argmax=False,  )
generate_based_sequence(length_sentence,  argmax=False,  )
get_vocab(file,  lower=False,  )
train_random_batch(  )
train_random_sequence(  )


mlmodels\model_tf\raw\tfcode2\RNN\Text-Generator\generate-shakespeare-lstm.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_length(length_sentence,  argmax=False,  )
generate_based_sequence(length_sentence,  argmax=False,  )
get_vocab(file,  lower=False,  )
train_random_batch(  )
train_random_sequence(  )


mlmodels\model_tf\raw\tfcode2\Seq-to-Seq\bahdanau-attention\bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,   )
check_accuracy(logits, Y,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tfcode2\Seq-to-Seq\Basic-Seq2Seq\helpers.py
----------------methods----------------

---------------functions---------------
batch(inputs,  max_sequence_length=None,  )
random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size,   )


mlmodels\model_tf\raw\tfcode2\Seq-to-Seq\beam\decoder-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
predict(X, Y, from_dict, to_dict, batch_size,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tfcode2\Seq-to-Seq\bidirectional\birnn.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
predict(X, Y, from_dict, to_dict, batch_size,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tfcode2\Seq-to-Seq\estimator\estimator.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, batch_size, from_dict_size, to_dict_size,  grad_clip=5.0,  )
Chatbot.lstm_cell(self,  reuse=False,  )
Chatbot.model_fn(self, features, labels, mode,   )
Chatbot.seq2seq(self, x_dict, reuse,   )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tfcode2\Seq-to-Seq\luong-attention\attention-basicdecoder.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
predict(X, Y, from_dict, to_dict, batch_size,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\attention\1.bahdanau.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Bahdanau.__call__(self, inputs, state,  scope=None,  )
Bahdanau.__init__(self, hidden_size, output_size, encoder_outputs,   )
Bahdanau.get_attention(self, inputs, state,   )
Bahdanau.output_size(self,   )
Bahdanau.reset_state(self,   )
Bahdanau.state_size(self,   )
Model.__init__(self, size_layer, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\attention\2.luong.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Luong.__call__(self, inputs, state,  scope=None,  )
Luong.__init__(self, hidden_size, output_size, encoder_outputs,   )
Luong.output_size(self,   )
Luong.reset_state(self,   )
Luong.state_size(self,   )
Model.__init__(self, size_layer, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\attention\3.hierarchical.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\attention\4.additive.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\attention\5.soft.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\attention\6.attention-over-attention.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, maxlen,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\attention\7.bahdanau-api.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\attention\8.luong-api.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\chatbot\1.basic-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\10.basic-birnn-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
check_accuracy(logits, Y,   )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\11.lstm-birnn-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\12.gru-birnn-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\13.basic-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\14.lstm-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\15.gru-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\16.basic-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\17.lstm-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\18.gru-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\19.lstm-birnn-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\2.lstm-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\20.gru-birnn-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\21.lstm-birnn-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\22.gru-birnn-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\23.lstm-birnn-seq2seq-bahdanau-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\24.gru-birnn-seq2seq-bahdanau-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\25.lstm-seq2seq-greedy-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\26.gru-seq2seq-greedy-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\27.lstm-seq2seq-greedy-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\28.gru-seq2seq-greedy-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\29.lstm-seq2seq-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\3.gru-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\30.gru-seq2seq-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\31.lstm-birnn-seq2seq-beam-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\32.gru-birnn-seq2seq-beam-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\33.lstm-birnn-seq2seq-luong-bahdanau-stack-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\34.gru-birnn-seq2seq-luong-bahdanau-stack-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\35.byte-net.py
----------------methods----------------
ByteNet.__init__(self, from_vocab_size, to_vocab_size, channels, encoder_dilations, decoder_dilations, encoder_filter_width, decoder_filter_width,  learning_rate=0.001, beta1=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width,  causal=True,  )
clean_text(text,   )
conv1d(input_, output_channels,  dilation=1, filter_width=1, causal=False,  )
layer_normalization(x,  epsilon=1e-8,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\36.estimator.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, batch_size, from_dict_size, to_dict_size,  grad_clip=5.0,  )
Chatbot.lstm_cell(self,  reuse=False,  )
Chatbot.model_fn(self, features, labels, mode,   )
Chatbot.seq2seq(self, x_dict, reuse,   )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\37.capsule-lstm-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, seq_len, maxlen, from_dict_size, to_dict_size, learning_rate, batch_size,  kernels=[2, 4, 4], strides=[3, 2, 1], epsilon=1e-8,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pad_sentence_batch_static(sentence_batch, pad_int,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\38.capsule-lstm-seq2seq-luong-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, seq_len, maxlen, from_dict_size, to_dict_size, learning_rate, batch_size,  kernels=[2, 4, 4], strides=[3, 2, 1], epsilon=1e-8, force_teaching_ratio=0.5, beam_width=5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pad_sentence_batch_static(sentence_batch, pad_int,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\39.lstm-birnn-seq2seq-luong-bahdanau-stack-beam-dropout-l2.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\4.basic-seq2seq-api-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\40.dnc-seq2seq-bahdanau-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  attn_input_feeding=True,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\41.lstm-birnn-seq2seq-beam-luongmonotic.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\42.lstm-birnn-seq2seq-beam-bahdanaumonotic.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\43.memory-network-basic.py
----------------methods----------------
QA.__init__(self, vocab_size_from, vocab_size_to, size_layer, learning_rate,  n_hops=3,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
embed_seq(x, vocab_size,  zero_pad=True,  )
hop_forward(memory_o, memory_i, response_proj, inputs_len, questions_len,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
position_encoding(sentence_size, embedding_size,   )
post_softmax_masking(x, seq_len,   )
pre_softmax_masking(x, seq_len,   )
quest_mem(x, vocab_size, max_quest_len,   )
shift_right(x,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\44.memory-network-lstm.py
----------------methods----------------
QA.__init__(self, vocab_size_from, vocab_size_to, size_layer, learning_rate,  n_hops=3,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
check_accuracy(logits, Y,   )
clean_text(text,   )
embed_seq(x, vocab_size,  zero_pad=True,  )
hop_forward(memory_o, memory_i, response_proj, inputs_len, questions_len,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
position_encoding(sentence_size, embedding_size,   )
post_softmax_masking(x, seq_len,   )
pre_softmax_masking(x, seq_len,   )
quest_mem(x, vocab_size, max_quest_len,   )
shift_right(x,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\45.attention-is-all-you-need.py
----------------methods----------------
Chatbot.__init__(self, size_layer, embedded_size, from_dict_size, to_dict_size, learning_rate,  num_blocks=2, num_heads=8, min_freq=50,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
label_smoothing(inputs,  epsilon=0.1,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_position_encoding(inputs, mask, embed_dim,   )
multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pointwise_feedforward(inputs, hidden_units,  activation=None,  )
sinusoidal_position_encoding(inputs, mask, repr_dim,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\46.transformer-xl.py
----------------methods----------------
Chatbot.__init__(self,   )
Chatbot._gather_logprob(logprob, target,   )
Chatbot._logit(x, W, b, proj,   )
Chatbot._logit(x, W, b, proj,   )

---------------functions---------------
_cache_mem(curr_out, prev_mem,  mem_len=None,  )
_create_mask(qlen, mlen,  same_length=False,  )
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
embedding_lookup(lookup_table, x,   )
mask_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, proj_same_dim=True, scope="adaptive_embed",  **kwargs)
mask_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs, params, tie_projs,  initializer=None, proj_initializer=None, div_val=1, scope="adaptive_softmax", proj_same_dim=True, return_mean=True,  **kwargs)
mul_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, perms=None, proj_same_dim=True, scope="adaptive_embed",  )
mul_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs, params, tie_projs,  initializer=None, proj_initializer=None, div_val=1, perms=None, proj_same_dim=True, scope="adaptive_softmax",  **kwargs)
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
positional_embedding(pos_seq, inv_freq,  bsz=None,  )
positionwise_FF(inp, d_model, d_inner, kernel_initializer,  scope="ff",  )
rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model, n_head, d_head, kernel_initializer,  scope="rel_attn",  )
rel_shift(x,   )
str_idx(corpus, dic,   )
transformer(dec_inp, mems, n_token, n_layer, d_model, d_embed, n_head, d_head, d_inner, initializer,  proj_initializer=None, mem_len=None, cutoffs=[], div_val=1, tie_projs=[], same_length=False, clamp_len=-1, untie_r=False, proj_same_dim=True, scope="transformer", reuse=tf.AUTO_REUSE,  )


mlmodels\model_tf\raw\tf_nlp\chatbot\47.attention-is-all-you-need-beam-search.py
----------------methods----------------
Chatbot.__init__(self, size_layer, embedded_size, from_dict_size, to_dict_size, learning_rate,  num_blocks=2, num_heads=8, min_freq=50,  )
Hypothesis.__init__(self, log_prob, seq,   )
Hypothesis.step(self,   )

---------------functions---------------
beam_search(batch_x, beam_size,  num_ans=5, normalize_by_len=1.0,  )
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
label_smoothing(inputs,  epsilon=0.1,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_position_encoding(inputs, mask, embed_dim,   )
multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pointwise_feedforward(inputs, hidden_units,  activation=None,  )
sinusoidal_position_encoding(inputs, mask, repr_dim,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\48.transformer-xl-lstm.py
----------------methods----------------
Chatbot.__init__(self,   )

---------------functions---------------
_cache_mem(curr_out, prev_mem,  mem_len=None,  )
_create_mask(qlen, mlen,  same_length=False,  )
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
embedding_lookup(lookup_table, x,   )
mask_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, proj_same_dim=True, scope="adaptive_embed",  **kwargs)
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
pad_sentence_batch_dynamic(sentence_batch, pad_int,   )
positional_embedding(pos_seq, inv_freq,  bsz=None,  )
positionwise_FF(inp, d_model, d_inner, kernel_initializer,  scope="ff",  )
rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model, n_head, d_head, kernel_initializer,  scope="rel_attn",  )
rel_shift(x,   )
str_idx(corpus, dic,   )
transformer(dec_inp, mems, n_token, n_layer, d_model, d_embed, n_head, d_head, d_inner, initializer,  proj_initializer=None, mem_len=None, cutoffs=[], div_val=1, tie_projs=[], same_length=False, clamp_len=-1, untie_r=False, proj_same_dim=True, scope="transformer", reuse=tf.AUTO_REUSE,  )


mlmodels\model_tf\raw\tf_nlp\chatbot\49.gpt-2-lstm.py
----------------methods----------------
Chatbot.__init__(self,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
pad_sentence_batch_dynamic(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\5.lstm-seq2seq-api-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\50.conv-encoder-conv-decoder.py
----------------methods----------------
Chatbot.__init__(self,   )
Hypothesis.__init__(self, log_prob, seq,   )
Hypothesis.step(self,   )

---------------functions---------------
beam_search(batch_x, beam_size,  num_ans=5, normalize_by_len=1.0,  )
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
decoder_block(inp, n_hidden, filter_size,   )
encoder_block(inp, n_hidden, filter_size,   )
glu(x,   )
layer(inp, conv_block, kernel_width, n_hidden,  residual=None,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\51.conv-encoder-lstm.py
----------------methods----------------
Chatbot.__init__(self,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
encoder_block(inp, n_hidden, filter_size,   )
glu(x,   )
layer(inp, conv_block, kernel_width, n_hidden,  residual=None,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\6.gru-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\7.basic-birnn-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\8.lstm-birnn-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\9.gru-birnn-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\access.py
----------------methods----------------
MemoryAccess.__init__(self,  memory_size=128, word_size=20, num_reads=1, num_writes=1, name="memory_access",  )
MemoryAccess._build(self, inputs, prev_state,   )
MemoryAccess._read_inputs(self, inputs,   )
MemoryAccess._read_weights(self, inputs, memory, prev_read_weights, link,   )
MemoryAccess._write_weights(self, inputs, memory, usage,   )
MemoryAccess.output_size(self,   )
MemoryAccess.state_size(self,   )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\addressing.py
----------------methods----------------
CosineWeights.__init__(self, num_heads, word_size,  strength_op=tf.nn.softplus, name="cosine_weights",  )
CosineWeights._build(self, memory, keys, strengths,   )
Freeness.__init__(self, memory_size,  name="freeness",  )
Freeness._allocation(self, usage,   )
Freeness._build(self, write_weights, free_gate, read_weights, prev_usage,   )
Freeness._usage_after_read(self, prev_usage, free_gate, read_weights,   )
Freeness._usage_after_write(self, prev_usage, write_weights,   )
Freeness.state_size(self,   )
Freeness.write_allocation_weights(self, usage, write_gates, num_writes,   )
TemporalLinkage.__init__(self, memory_size, num_writes,  name="temporal_linkage",  )
TemporalLinkage._build(self, write_weights, prev_state,   )
TemporalLinkage._link(self, prev_link, prev_precedence_weights, write_weights,   )
TemporalLinkage._precedence_weights(self, prev_precedence_weights, write_weights,   )
TemporalLinkage.directional_read_weights(self, link, prev_read_weights, forward,   )
TemporalLinkage.state_size(self,   )

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\dnc.py
----------------methods----------------
DNC.__init__(self, access_config, controller_config, output_size,  clip_value=None, name="dnc",  )
DNC._build(self, inputs, prev_state,   )
DNC._clip_if_enabled(self, x,   )
DNC.initial_state(self, batch_size,  dtype=tf.float32,  )
DNC.output_size(self,   )
DNC.state_size(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\chatbot\gpt_2.py
----------------methods----------------

---------------functions---------------
attention_mask(nd, ns,   )
attn(x, scope, n_state,   )
block(x, scope,   )
conv1d(x, scope, nf,   )
expand_tile(value, size,   )
gelu(x,   )
merge_states(x,   )
mlp(x, scope, n_state,   )
model(hparams, X,  past=None, scope="model", reuse=False,  )
norm(x, scope,   )
past_shape(  )
positions_for(tokens, past_length,   )
shape_list(x,   )
softmax(x,  axis=-1,  )
split_states(x, n,   )


mlmodels\model_tf\raw\tf_nlp\chatbot\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\bidirectional-rnn-vector.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\cnn-rnn-vector.py
----------------methods----------------
Model.__init__(self, sequence_length, dimension_input, dimension_output, learning_rate, filter_sizes, pooling_size, out_dimension, num_layer,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\cnn-vector.py
----------------methods----------------
Model.__init__(self, sequence_length, dimension_input, dimension_output, learning_rate, filter_sizes, out_dimension,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\feedforward-vector.py
----------------methods----------------
Model.__init__(self, dimension_input, size_layer, dimension_output, learning_rate,   )
Model_vec.__init__(self, batch_size, dimension_size, learning_rate, vocabulary_size,   )

---------------functions---------------
build_dataset(words, vocabulary_size,   )
clearstring(string,   )
generate_batch_skipgram(words, batch_size, num_skips, skip_window,   )
generatevector(dimension, batch_size, skip_size, skip_window, num_skips, iteration, words_real,   )
read_data(  )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\kmax-conv-vector.py
----------------methods----------------
Model.__init__(self, seq_len, dimension_input, dimension_output, learning_rate,  top_k=5, n_filters=250,  )
Model.add_kmax_pooling(self, x,   )

---------------functions---------------
add_conv1d(x, n_filters, kernel_size,  strides=1,  )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\LNLSTM-vector.py
----------------methods----------------
LN_LSTMCell.__init__(self, num_units,  f_bias=1.0, use_zoneout=False, zoneout_keep_h=0.9, zoneout_keep_c=0.5, is_training=True, reuse=None, name=None,  )
LN_LSTMCell.build(self, inputs_shape,   )
LN_LSTMCell.call(self, x, state,   )
LN_LSTMCell.output_size(self,   )
LN_LSTMCell.state_size(self,   )
LN_LSTMCell.zero_state(self, batch_size, dtype,   )
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------
layer_norm(x,  scope="layer_norm", alpha_start=1.0, bias_start=0.0,  )
layer_norm_all(h, base, num_units, scope,   )
moments_for_layer_norm(x,  axes=1, name=None,  )
zoneout(new_h, new_c, h, c, h_keep, c_keep, is_training,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\multihead-attention.py
----------------methods----------------
Model.__init__(self, dimension_input, dimension_output, seq_len, learning_rate,  num_heads=8, attn_windows=range(1, 6),  )
Model.multihead_attn(self, inputs, masks,   )
Model.window_mask(self, h_w,   )

---------------functions---------------
embed_seq(inputs,  vocab_size=None, embed_dim=None, zero_pad=False, scale=False,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_positional_encoding(inputs, embed_dim,  zero_pad=False, scale=False,  )
pointwise_feedforward(inputs,  num_units=[None, None], activation=None,  )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\ntm-vector.py
----------------methods----------------
Model.__init__(self, seq_len, size_layer, batch_size, dimension_input, dimension_output, learning_rate, memory_size, memory_vector_size,  read_head_num=4, write_head_num=1,  )
NTMCell.__call__(self, x, prev_state,   )
NTMCell.__init__(self, rnn_size, memory_size, memory_vector_dim, read_head_num, write_head_num,  addressing_mode="content_and_location", shift_range=1, reuse=False, output_dim=None,  )
NTMCell.addressing(self, k, beta, g, s, gamma, prev_M, prev_w,   )
NTMCell.zero_state(self, batch_size, dtype,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\only-attention-vector.py
----------------methods----------------
Model.__init__(self, seq_len, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------
sinusoidal_positional_encoding(inputs, num_units,  zero_pad=False, scale=False,  )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\rnn-attention-vector.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, attention_size, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\rnn-timestamp.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\rnn-vector-hinge.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\rnn-vector-stack.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\rnn-vector.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\self-optimized-feedforward-timestamp.py
----------------methods----------------
neuralnet.__init__(self, timestamp, num_hidden, size_layer,  learning_rate=0.01,  )

---------------functions---------------
clearstring(string,   )
generate_nn(timestamp, num_hidden, size_layer,   )
neural_network(timestamp, num_hidden, size_layer,  learning_rate=0.001, batch_size=200, epoch=20,  )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\seq2seq-vector-stable.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Deep-learning\seq2seq-vector.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Ensemble\featuring-ensemble.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Ensemble\oracle.py
----------------methods----------------

---------------functions---------------
basic_tokenize(tweet,   )
clearstring(string,   )
get_metric(vectorizer, X_raw, y_raw, name,   )
get_metric_oracle(X_raw, y_raw, vectorizers,   )
make_skip_tokenize(n, k,  include_all=True,  )
separate_dataset(trainset,   )
skipgram_tokenize(tweet,  n=None, k=None, include_all=True,  )
tokenize(tweet,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\LGB\lgb-tfidf-svd50.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\LGB\lgb-tfidf.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\LGB\lgb-timestamp.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\LGB\nce-vector-lgb.py
----------------methods----------------
Model_vec.__init__(self, bow_shape, batch_size, dimension_size, learning_rate, vocabulary_size, boundary,   )

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\Naive-Bayes\Bayes classifier.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\NB-SVM\NB-SVM.py
----------------methods----------------
NB_SVM.__init__(self,  C=1.0, dual=False, n_jobs=1,  )
NB_SVM.fit(self, x, y,   )
NB_SVM.predict(self, x,   )
NB_SVM.predict_proba(self, x,   )

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\preparation\prepare-dataset.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
read_data(location,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\preparation\prepare-vocab.py
----------------methods----------------

---------------functions---------------
build_vocab(words, n_words,   )
clearstring(string,   )
read_data(location,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\preparation\word-vector.py
----------------methods----------------

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
generate_batch(batch_size, num_skips, skip_window,   )
read_data(  )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\SVM\SVM.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\XGB\xgb-bow.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\XGB\xgb-tfidf-svd50.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\XGB\xgb-tfidf.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\XGB\xgb-timestamp-avg.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\Classification Comparison\XGB\xgb-timestamp50.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
separate_dataset(trainset,   )


mlmodels\model_tf\raw\tf_nlp\dependency-parser\1.birnn-bahdanau.py
----------------methods----------------
Model.__init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers, maxlen,   )

---------------functions---------------
generate_char_seq(batch,  UNK=2,  )
process_corpus(corpus,  until=None,  )


mlmodels\model_tf\raw\tf_nlp\dependency-parser\2.birnn-luong.py
----------------methods----------------
Model.__init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers, maxlen,   )

---------------functions---------------
generate_char_seq(batch,  UNK=2,  )
process_corpus(corpus,  until=None,  )


mlmodels\model_tf\raw\tf_nlp\dependency-parser\3.residual-network-bahdanau.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Model.__init__(self, dict_size, size_layers, learning_rate, maxlen,  num_blocks=3, block_size=128,  )

---------------functions---------------
process_corpus(corpus,  until=None,  )


mlmodels\model_tf\raw\tf_nlp\dependency-parser\4.residual-network-bahdanau-char.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Model.__init__(self, dict_size, char_dict_size, size_layers, learning_rate, maxlen,  num_blocks=3, block_size=64,  )

---------------functions---------------
generate_char_seq(batch, maxlen_c, maxlen,  UNK=2,  )
process_corpus(corpus,  until=None,  )


mlmodels\model_tf\raw\tf_nlp\embedded\1.cbow-softmax.py
----------------methods----------------
CBOW.__init__(self, sample_size, vocab_size, embedded_size,  window_size=3,  )

---------------functions---------------
get_x(words, idx,   )
make_xy(int_words,   )


mlmodels\model_tf\raw\tf_nlp\embedded\10.fast-text.py
----------------methods----------------
Model.__init__(self, graph_params,   )
Model.train(self, train, epoch, batch_size,   )

---------------functions---------------
_pad_sequence(sequence, n,  pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None,  )
build_dict(word_counter,  vocab_size=50000,  )
build_training_set(word_array,  maxlen=100,  )
build_word_array(sentences, vocab_size,   )
counter_words(sentences,   )
doc2num(word_list, dictionary,   )
doc2num(word_list, dictionary,  ngrams=(2, 3),  )
generator(word,  ngram=(2, 3),  )
ngrams(sequence, n,  pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None,  )


mlmodels\model_tf\raw\tf_nlp\embedded\11.elmo.py
----------------methods----------------
BidirectionalLMDataset.__init__(self, string, vocab,   )
BidirectionalLMDataset.iter_batches(self, batch_size, num_steps,   )
LMDataset.__init__(self, string, vocab,  reverse=False,  )
LMDataset._load_string(self, string,   )
LMDataset.get_sentence(self,   )
LMDataset.iter_batches(self, batch_size, num_steps,   )
LMDataset.max_word_length(self,   )
LMDataset.vocab(self,   )
LanguageModel.__init__(self, options, is_training,   )
LanguageModel._build(self,   )
LanguageModel._build_loss(self, lstm_outputs,   )
LanguageModel._build_word_char_embeddings(self,   )
LanguageModel._build_word_embeddings(self,   )
UnicodeCharsVocabulary.__init__(self, dictionary, rev_dictionary, max_word_length,   **kwargs)
UnicodeCharsVocabulary._convert_word_to_char_ids(self, word,   )
UnicodeCharsVocabulary.encode_chars(self, sentence,  reverse=False, split=True,  )
UnicodeCharsVocabulary.max_word_length(self,   )
UnicodeCharsVocabulary.word_char_ids(self,   )
UnicodeCharsVocabulary.word_to_char_ids(self, word,   )
Vocabulary.__init__(self, dictionary, rev_dictionary,   )
Vocabulary.decode(self, cur_ids,   )
Vocabulary.encode(self, sentence,  reverse=False, split=True,  )
Vocabulary.end_string(self,   )
Vocabulary.id_to_word(self, cur_id,   )
Vocabulary.size(self,   )
Vocabulary.start_string(self,   )
Vocabulary.unk(self,   )
Vocabulary.word_to_id(self, word,   )

---------------functions---------------
_get_batch(generator, batch_size, num_steps, max_word_length,   )
_get_feed_dict_from_X(X, model, char_inputs, bidirectional,   )
build_dataset(words, n_words,  atleast=1,  )


mlmodels\model_tf\raw\tf_nlp\embedded\2.cbow-nce.py
----------------methods----------------
CBOW.__init__(self, sample_size, vocab_size, embedded_size,  window_size=3,  )

---------------functions---------------
get_x(words, idx,   )
make_xy(int_words,   )


mlmodels\model_tf\raw\tf_nlp\embedded\3.skipgram-softmax.py
----------------methods----------------
SKIPGRAM.__init__(self, sample_size, vocab_size, embedded_size,  window_size=3,  )

---------------functions---------------
get_y(words, idx,   )
make_xy(int_words,   )


mlmodels\model_tf\raw\tf_nlp\embedded\4.skipgram-nce.py
----------------methods----------------
SKIPGRAM.__init__(self, sample_size, vocab_size, embedded_size,  window_size=3,  )

---------------functions---------------
get_y(words, idx,   )
make_xy(int_words,   )


mlmodels\model_tf\raw\tf_nlp\embedded\5.lda2vec.py
----------------methods----------------
LDA2VEC.__init__(self, num_unique_documents, vocab_size, num_topics, freqs,  embedding_size=128, num_sampled=40, learning_rate=1e-3, lmbda=150.0, alpha=None, power=0.75, batch_size=32, clip_gradients=5.0,  **kwargs)
LDA2VEC.train(self, pivot_words, target_words, doc_ids, num_epochs,  switch_loss=3,  )

---------------functions---------------
skipgrams(sequence, vocabulary_size,  window_size=4, negative_samples=1.0, shuffle=True, categorical=False, sampling_table=None, seed=None,  )


mlmodels\model_tf\raw\tf_nlp\embedded\6.supervised-embedded.py
----------------methods----------------
CBOW.__init__(self, sample_size, vocab_size, embedded_size,  window_size=3,  )
Embedded.__init__(self, document_size, embedded_size, embedded_words,  sample_size=32, rnn_size=128, num_layers=2,  )

---------------functions---------------
clearstring(string,   )
get_x(words, idx,   )
make_xy(int_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
separate_dataset(trainset,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\embedded\7.triplet-loss.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output,  margin=0.2,  )

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
compute_euclidean_distance(x, y,   )
compute_triplet_loss(anchor_feature, positive_feature, negative_feature,  margin=0.01,  )
get_one_triplet(input_data, input_labels, n_labels,   )
separate_dataset(trainset,   )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_nlp\embedded\8.auto-encoder.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, seq_len,   )

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
separate_dataset(trainset,   )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_nlp\embedded\9.batch-all-triplet-loss-lstm-embedded.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output,   )

---------------functions---------------
_get_anchor_negative_triplet_mask(labels,   )
_get_anchor_positive_triplet_mask(labels,   )
_get_triplet_mask(labels,   )
_pairwise_distances(embeddings,  squared=False,  )
batch_all_triplet_loss(labels, embeddings, margin,  squared=False,  )


mlmodels\model_tf\raw\tf_nlp\entity-tagging\1.concat-bahdanau.py
----------------methods----------------
Model.__init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers,   )

---------------functions---------------
char_str_idx(corpus, dic,  UNK=0,  )
freeze_graph(model_dir, output_node_names,   )
generate_char_seq(batch,   )
generate_char_seq(batch, idx2word, char2idx,   )
iter_seq(x,   )
load_graph(frozen_graph_filename,   )
parse_XY(texts, labels,   )
parse_raw(filename,   )
pred2label(pred,   )
process_string(string,   )
to_title(string,   )
to_train_seq(  *args)


mlmodels\model_tf\raw\tf_nlp\entity-tagging\2.concat-luong.py
----------------methods----------------
Model.__init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers,   )

---------------functions---------------
char_str_idx(corpus, dic,  UNK=0,  )
freeze_graph(model_dir, output_node_names,   )
generate_char_seq(batch,   )
generate_char_seq(batch, idx2word, char2idx,   )
iter_seq(x,   )
load_graph(frozen_graph_filename,   )
parse_XY(texts, labels,   )
parse_raw(filename,   )
pred2label(pred,   )
process_string(string,   )
to_title(string,   )
to_train_seq(  *args)


mlmodels\model_tf\raw\tf_nlp\entity-tagging\3.concat.py
----------------methods----------------
Model.__init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers,   )

---------------functions---------------
char_str_idx(corpus, dic,  UNK=0,  )
freeze_graph(model_dir, output_node_names,   )
generate_char_seq(batch,   )
generate_char_seq(batch, idx2word, char2idx,   )
iter_seq(x,   )
load_graph(frozen_graph_filename,   )
parse_XY(texts, labels,   )
parse_raw(filename,   )
pred2label(pred,   )
process_string(string,   )
to_title(string,   )
to_train_seq(  *args)


mlmodels\model_tf\raw\tf_nlp\entity-tagging\4.concat-bahdanau-char-ngrams.py
----------------methods----------------
Model.__init__(self, dim_word, dropout, learning_rate, hidden_size_word, num_layers,   )

---------------functions---------------
_pad_sequence(sequence, n,  pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None,  )
generate_char_seq(batch,   )
generate_char_seq(batch,  UNK=2,  )
get_ngrams(s,  grams=(2, 3, 4),  )
iter_seq(x,   )
ngrams(sequence, n,  pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None,  )
parse_XY(texts, labels,   )
parse_raw(filename,   )
process_string(string,   )
to_title(string,   )
to_train_seq(  *args)


mlmodels\model_tf\raw\tf_nlp\entity-tagging\5.residual-char-ngrams.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Model.__init__(self, dict_size, size_layers, learning_rate, maxlen,  num_blocks=3, block_size=128,  )

---------------functions---------------
_pad_sequence(sequence, n,  pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None,  )
generate_char_seq(batch, maxlen,   )
get_ngrams(s,  grams=(2, 3, 4),  )
iter_seq(x,   )
ngrams(sequence, n,  pad_left=False, pad_right=False, left_pad_symbol=None, right_pad_symbol=None,  )
parse_XY(texts, labels,   )
parse_raw(filename,   )
process_string(string,   )
to_title(string,   )
to_train_seq(  *args)


mlmodels\model_tf\raw\tf_nlp\generator\1.char-generator-lstm.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_sequence(length_sentence,  argmax=False,  )
get_vocab(file,  lower=False,  )
train_random_sequence(  )


mlmodels\model_tf\raw\tf_nlp\generator\10.gru-seq2seq-beam-word.py
----------------methods----------------
Generator.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
generate_based_sequence(length_sentence,   )
train_random_batch(  )


mlmodels\model_tf\raw\tf_nlp\generator\11.gru-seq2seq-bahdanau-greedy-char.py
----------------methods----------------
Generator.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
generate_based_sequence(length_sentence,   )
train_random_batch(  )


mlmodels\model_tf\raw\tf_nlp\generator\12.gru-seq2seq-bahdanau-greedy-word.py
----------------methods----------------
Generator.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
generate_based_sequence(length_sentence,   )
train_random_batch(  )


mlmodels\model_tf\raw\tf_nlp\generator\2.char-rnn-beam.py
----------------methods----------------
Model.__init__(self, seq_len, vocab_size, hidden_dim,   )

---------------functions---------------
cell_fn(  )
clip_grads(loss,   )
end_sentence(x,   )
multi_cell_fn(  )
parse_text(file_path,   )
start_sentence(x,   )


mlmodels\model_tf\raw\tf_nlp\generator\3.char-generator-lstm-embedding.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_sequence(length_sentence,  argmax=False,  )
get_vocab(file,  lower=False,  )
train_random_batch(  )
train_random_sequence(  )


mlmodels\model_tf\raw\tf_nlp\generator\4.word-generator-lstm.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_sequence(length_sentence,  argmax=False,  )
get_vocab(file,  lower=False,  )
train_random_sequence(  )


mlmodels\model_tf\raw\tf_nlp\generator\5.word-generator-lstm-embedding.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_sequence(length_sentence,  argmax=False,  )
get_vocab(file,  lower=False,  )
train_random_sequence(  )


mlmodels\model_tf\raw\tf_nlp\generator\6.gru-seq2seq-greedy-char.py
----------------methods----------------
Generator.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
generate_based_sequence(length_sentence,   )
train_random_batch(  )


mlmodels\model_tf\raw\tf_nlp\generator\7.gru-seq2seq-greedy-word.py
----------------methods----------------
Generator.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
generate_based_sequence(length_sentence,   )
train_random_batch(  )


mlmodels\model_tf\raw\tf_nlp\generator\8.char-generator-lstm-bahdanau.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_sequence(length_sentence,  argmax=False,  )
get_vocab(file,  lower=False,  )
train_random_sequence(  )


mlmodels\model_tf\raw\tf_nlp\generator\9.char-generator-lstm-luong.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension, sequence_length, learning_rate,   )

---------------functions---------------
embed_to_onehot(data, vocab,   )
generate_based_sequence(length_sentence,  argmax=False,  )
get_vocab(file,  lower=False,  )
train_random_sequence(  )


mlmodels\model_tf\raw\tf_nlp\language-detection\1.fast-text-ngrams.py
----------------methods----------------
Model.__init__(self, learning_rate,   )

---------------functions---------------
clean_text(string,   )
convert_sparse_matrix_to_sparse_tensor(X,  limit=5,  )


mlmodels\model_tf\raw\tf_nlp\misc\1.attention-visualization-bahdanau.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\misc\2.attention-visualization-luong.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\1.basic-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\10.basic-birnn-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\11.lstm-birnn-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\12.gru-birnn-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\13.basic-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\14.lstm-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\15.gru-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\16.basic-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\17.lstm-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\18.gru-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\19.lstm-birnn-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\2.lstm-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\20.gru-birnn-seq2seq-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\21.lstm-birnn-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\22.gru-birnn-seq2seq-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\23.lstm-birnn-seq2seq-bahdanau-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\24.gru-birnn-seq2seq-bahdanau-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\25.lstm-seq2seq-greedy-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\26.gru-seq2seq-greedy-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\27.lstm-seq2seq-greedy-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\28.gru-seq2seq-greedy-bahdanau.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\29.lstm-seq2seq-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\3.gru-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\30.gru-seq2seq-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\31.lstm-birnn-seq2seq-beam-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\32.gru-birnn-seq2seq-beam-luong.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\33.lstm-birnn-seq2seq-luong-bahdanau-stack-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\34.gru-birnn-seq2seq-luong-bahdanau-stack-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\35.byte-net.py
----------------methods----------------
ByteNet.__init__(self, from_vocab_size, to_vocab_size, channels, encoder_dilations, decoder_dilations, encoder_filter_width, decoder_filter_width,  learning_rate=0.001, beta1=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width,  causal=True,  )
conv1d(input_, output_channels,  dilation=1, filter_width=1, causal=False,  )
layer_normalization(x,  epsilon=1e-8,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\36.estimator.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, batch_size, from_dict_size, to_dict_size,  grad_clip=5.0,  )
Chatbot.lstm_cell(self,  reuse=False,  )
Chatbot.model_fn(self, features, labels, mode,   )
Chatbot.seq2seq(self, x_dict, reuse,   )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\37.capsule-lstm-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, seq_len, maxlen, from_dict_size, to_dict_size, learning_rate, batch_size,  kernels=[2, 4, 4], strides=[3, 2, 1], epsilon=1e-8,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pad_sentence_batch_static(sentence_batch, pad_int,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\38.capsule-lstm-seq2seq-luong-beam.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, seq_len, maxlen, from_dict_size, to_dict_size, learning_rate, batch_size,  kernels=[2, 4, 4], strides=[3, 2, 1], epsilon=1e-8, force_teaching_ratio=0.5, beam_width=5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
clean_text(text,   )
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pad_sentence_batch_static(sentence_batch, pad_int,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\39.lstm-birnn-seq2seq-luong-bahdanau-stack-beam-dropout-l2.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\4.basic-seq2seq-api-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\40.dnc-seq2seq-bahdanau-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  attn_input_feeding=True,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\41.lstm-birnn-seq2seq-beam-luongmonotic.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\42.lstm-birnn-seq2seq-beam-bahdanaumonotic.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\43.memory-network-basic.py
----------------methods----------------
QA.__init__(self, vocab_size_from, vocab_size_to, size_layer, learning_rate,  n_hops=3,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
embed_seq(x, vocab_size,  zero_pad=True,  )
hop_forward(memory_o, memory_i, response_proj, inputs_len, questions_len,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
position_encoding(sentence_size, embedding_size,   )
post_softmax_masking(x, seq_len,   )
pre_softmax_masking(x, seq_len,   )
quest_mem(x, vocab_size, max_quest_len,   )
shift_right(x,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\44.memory-network-lstm.py
----------------methods----------------
QA.__init__(self, vocab_size_from, vocab_size_to, size_layer, learning_rate,  n_hops=3,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
embed_seq(x, vocab_size,  zero_pad=True,  )
hop_forward(memory_o, memory_i, response_proj, inputs_len, questions_len,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
position_encoding(sentence_size, embedding_size,   )
post_softmax_masking(x, seq_len,   )
pre_softmax_masking(x, seq_len,   )
quest_mem(x, vocab_size, max_quest_len,   )
shift_right(x,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\45.attention-is-all-you-need.py
----------------methods----------------
Chatbot.__init__(self, size_layer, embedded_size, from_dict_size, to_dict_size, learning_rate,  num_blocks=2, num_heads=8, min_freq=50,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
label_smoothing(inputs,  epsilon=0.1,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_position_encoding(inputs, mask, embed_dim,   )
multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pointwise_feedforward(inputs, hidden_units,  activation=None,  )
sinusoidal_position_encoding(inputs, mask, repr_dim,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\46.transformer-xl.py
----------------methods----------------
Chatbot.__init__(self,   )
Chatbot._gather_logprob(logprob, target,   )
Chatbot._logit(x, W, b, proj,   )
Chatbot._logit(x, W, b, proj,   )

---------------functions---------------
_cache_mem(curr_out, prev_mem,  mem_len=None,  )
_create_mask(qlen, mlen,  same_length=False,  )
build_dataset(words, n_words,  atleast=1,  )
embedding_lookup(lookup_table, x,   )
mask_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, proj_same_dim=True, scope="adaptive_embed",  **kwargs)
mask_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs, params, tie_projs,  initializer=None, proj_initializer=None, div_val=1, scope="adaptive_softmax", proj_same_dim=True, return_mean=True,  **kwargs)
mul_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, perms=None, proj_same_dim=True, scope="adaptive_embed",  )
mul_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs, params, tie_projs,  initializer=None, proj_initializer=None, div_val=1, perms=None, proj_same_dim=True, scope="adaptive_softmax",  **kwargs)
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
positional_embedding(pos_seq, inv_freq,  bsz=None,  )
positionwise_FF(inp, d_model, d_inner, kernel_initializer,  scope="ff",  )
rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model, n_head, d_head, kernel_initializer,  scope="rel_attn",  )
rel_shift(x,   )
str_idx(corpus, dic,   )
transformer(dec_inp, mems, n_token, n_layer, d_model, d_embed, n_head, d_head, d_inner, initializer,  proj_initializer=None, mem_len=None, cutoffs=[], div_val=1, tie_projs=[], same_length=False, clamp_len=-1, untie_r=False, proj_same_dim=True, scope="transformer", reuse=tf.AUTO_REUSE,  )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\47.attention-is-all-you-need-beam-search.py
----------------methods----------------
Chatbot.__init__(self, size_layer, embedded_size, from_dict_size, to_dict_size, learning_rate,  num_blocks=2, num_heads=8, min_freq=50,  )
Hypothesis.__init__(self, log_prob, seq,   )
Hypothesis.step(self,   )

---------------functions---------------
beam_search(batch_x, beam_size,  num_ans=5, normalize_by_len=1.0,  )
build_dataset(words, n_words,  atleast=1,  )
label_smoothing(inputs,  epsilon=0.1,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_position_encoding(inputs, mask, embed_dim,   )
multihead_attn(queries, keys, q_masks, k_masks, future_binding, num_units, num_heads,   )
pad_sentence_batch(sentence_batch, pad_int,   )
pointwise_feedforward(inputs, hidden_units,  activation=None,  )
sinusoidal_position_encoding(inputs, mask, repr_dim,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\48.conv-encoder-conv-decoder.py
----------------methods----------------
Chatbot.__init__(self,   )
Hypothesis.__init__(self, log_prob, seq,   )
Hypothesis.step(self,   )

---------------functions---------------
beam_search(batch_x, beam_size,  num_ans=5, normalize_by_len=1.0,  )
build_dataset(words, n_words,  atleast=1,  )
decoder_block(inp, n_hidden, filter_size,   )
encoder_block(inp, n_hidden, filter_size,   )
glu(x,   )
layer(inp, conv_block, kernel_width, n_hidden,  residual=None,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\49.conv-encoder-lstm.py
----------------methods----------------
Chatbot.__init__(self,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
encoder_block(inp, n_hidden, filter_size,   )
glu(x,   )
layer(inp, conv_block, kernel_width, n_hidden,  residual=None,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\5.lstm-seq2seq-api-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\50.byte-net-greedy.py
----------------methods----------------
ByteNet.__init__(self, from_vocab_size, to_vocab_size, channels, encoder_dilations, decoder_dilations, encoder_filter_width, decoder_filter_width,  learning_rate=0.001, beta1=0.5,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width, block_type,  causal=True, reuse=False,  )
conv1d(input_, output_channels, block_name, reuse,  dilation=1, filter_width=1, causal=False,  )
layer_normalization(inputs, block_name, reuse,  epsilon=1e-8,  )
pad_sentence_batch(sentence_batch, sentence_batch_y, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\6.gru-seq2seq-greedy.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\7.basic-birnn-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\8.lstm-birnn-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\9.gru-birnn-seq2seq-manual.py
----------------methods----------------
Chatbot.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\access.py
----------------methods----------------
MemoryAccess.__init__(self,  memory_size=128, word_size=20, num_reads=1, num_writes=1, name="memory_access",  )
MemoryAccess._build(self, inputs, prev_state,   )
MemoryAccess._read_inputs(self, inputs,   )
MemoryAccess._read_weights(self, inputs, memory, prev_read_weights, link,   )
MemoryAccess._write_weights(self, inputs, memory, usage,   )
MemoryAccess.output_size(self,   )
MemoryAccess.state_size(self,   )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\addressing.py
----------------methods----------------
CosineWeights.__init__(self, num_heads, word_size,  strength_op=tf.nn.softplus, name="cosine_weights",  )
CosineWeights._build(self, memory, keys, strengths,   )
Freeness.__init__(self, memory_size,  name="freeness",  )
Freeness._allocation(self, usage,   )
Freeness._build(self, write_weights, free_gate, read_weights, prev_usage,   )
Freeness._usage_after_read(self, prev_usage, free_gate, read_weights,   )
Freeness._usage_after_write(self, prev_usage, write_weights,   )
Freeness.state_size(self,   )
Freeness.write_allocation_weights(self, usage, write_gates, num_writes,   )
TemporalLinkage.__init__(self, memory_size, num_writes,  name="temporal_linkage",  )
TemporalLinkage._build(self, write_weights, prev_state,   )
TemporalLinkage._link(self, prev_link, prev_precedence_weights, write_weights,   )
TemporalLinkage._precedence_weights(self, prev_precedence_weights, write_weights,   )
TemporalLinkage.directional_read_weights(self, link, prev_read_weights, forward,   )
TemporalLinkage.state_size(self,   )

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\dnc.py
----------------methods----------------
DNC.__init__(self, access_config, controller_config, output_size,  clip_value=None, name="dnc",  )
DNC._build(self, inputs, prev_state,   )
DNC._clip_if_enabled(self, x,   )
DNC.initial_state(self, batch_size,  dtype=tf.float32,  )
DNC.output_size(self,   )
DNC.state_size(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\gpt_2.py
----------------methods----------------

---------------functions---------------
attention_mask(nd, ns,   )
attn(x, scope, n_state,   )
block(x, scope,   )
conv1d(x, scope, nf,   )
expand_tile(value, size,   )
gelu(x,   )
merge_states(x,   )
mlp(x, scope, n_state,   )
model(hparams, X,  past=None, scope="model", reuse=False,  )
norm(x, scope,   )
past_shape(  )
positions_for(tokens, past_length,   )
shape_list(x,   )
softmax(x,  axis=-1,  )
split_states(x, n,   )


mlmodels\model_tf\raw\tf_nlp\neural-machine-translation\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )


mlmodels\model_tf\raw\tf_nlp\not-deep-learning\decomposition-summarization\1.lda.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\not-deep-learning\decomposition-summarization\2.lsa.py
----------------methods----------------
LSA.__init__(self, tfidf,   )
LSA._calc_svd(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\not-deep-learning\decomposition-summarization\3.nmf.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\not-deep-learning\markov-chatbot\markov-chatbot.py
----------------methods----------------

---------------functions---------------
clean_text(text,   )
generate_reply(lm, order,  nletters=1000,  )
generate_word(lm, history, order,   )
train_chatbot(data,  order=4,  )


mlmodels\model_tf\raw\tf_nlp\ocr\1.cnn-rnn-lstm\cnn-lstm-ctc.py
----------------methods----------------
Model.__init__(self,   )
Model.batch_norm(self, name, x,   )
Model.conv2d(self, x, name, filter_size, channel_in, channel_out, strides,   )
Model.leaky_relu(self, x,  leak=0,  )
Model.max_pool(self, x, size, strides,   )

---------------functions---------------
accuracy_calculation(original_seq, decoded_seq,  ignore_value=-1,  )
sparse_tuple_from_label(sequences,  dtype=np.int32,  )


mlmodels\model_tf\raw\tf_nlp\pos-tagging\1.concat-bahdanau.py
----------------methods----------------
Model.__init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers,   )

---------------functions---------------
char_str_idx(corpus, dic,  UNK=0,  )
freeze_graph(model_dir, output_node_names,   )
generate_char_seq(batch,   )
generate_char_seq(batch, idx2word, char2idx,   )
iter_seq(x,   )
load_graph(frozen_graph_filename,   )
parse_XY(texts, labels,   )
pred2label(pred,   )
process_string(string,   )
to_title(string,   )
to_train_seq(  *args)


mlmodels\model_tf\raw\tf_nlp\pos-tagging\2.concat-luong.py
----------------methods----------------
Model.__init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers,   )

---------------functions---------------
char_str_idx(corpus, dic,  UNK=0,  )
freeze_graph(model_dir, output_node_names,   )
generate_char_seq(batch,   )
generate_char_seq(batch, idx2word, char2idx,   )
iter_seq(x,   )
load_graph(frozen_graph_filename,   )
parse_XY(texts, labels,   )
pred2label(pred,   )
process_string(string,   )
to_title(string,   )
to_train_seq(  *args)


mlmodels\model_tf\raw\tf_nlp\pos-tagging\3.concat.py
----------------methods----------------
Model.__init__(self, dim_word, dim_char, dropout, learning_rate, hidden_size_char, hidden_size_word, num_layers,   )

---------------functions---------------
char_str_idx(corpus, dic,  UNK=0,  )
freeze_graph(model_dir, output_node_names,   )
generate_char_seq(batch,   )
generate_char_seq(batch, idx2word, char2idx,   )
iter_seq(x,   )
load_graph(frozen_graph_filename,   )
parse_XY(texts, labels,   )
pred2label(pred,   )
process_string(string,   )
to_title(string,   )
to_train_seq(  *args)


mlmodels\model_tf\raw\tf_nlp\question-answer\1.end-to-end-basic.py
----------------methods----------------
BaseDataLoader.__init__(self,   )
DataLoader.__init__(self, path, is_training,  vocab=None, params=None,  )
DataLoader.build_vocab(self, data,   )
DataLoader.load_data(self, path,   )
DataLoader.padding(self, data, lens,   )
QA.__init__(self, vocab_size,   )

---------------functions---------------
bAbI_data_load(path,  END=["<end>"],  )
embed_seq(x, vocab_size,  zero_pad=True,  )
hop_forward(question, memory_o, memory_i, response_proj, inputs_len, questions_len, is_training,   )
input_mem(x, vocab_size, max_sent_len, is_training,   )
position_encoding(sentence_size, embedding_size,   )
post_softmax_masking(x, seq_len,   )
pre_softmax_masking(x, seq_len,   )
quest_mem(x, vocab_size, max_quest_len, is_training,   )
shift_right(x,   )


mlmodels\model_tf\raw\tf_nlp\question-answer\2.end-to-end-gru.py
----------------methods----------------
BaseDataLoader.__init__(self,   )
DataLoader.__init__(self, path, is_training,  vocab=None, params=None,  )
DataLoader.build_vocab(self, data,   )
DataLoader.load_data(self, path,   )
DataLoader.padding(self, data, lens,   )
QA.__init__(self, vocab_size,   )

---------------functions---------------
bAbI_data_load(path,  END=["<end>"],  )
embed_seq(x, vocab_size,  zero_pad=True,  )
hop_forward(question, memory_o, memory_i, response_proj, inputs_len, questions_len, is_training,   )
input_mem(x, vocab_size, max_sent_len, is_training,   )
position_encoding(sentence_size, embedding_size,   )
post_softmax_masking(x, seq_len,   )
pre_softmax_masking(x, seq_len,   )
quest_mem(x, vocab_size, max_quest_len, is_training,   )
shift_right(x,   )


mlmodels\model_tf\raw\tf_nlp\question-answer\3.end-to-end-lstm.py
----------------methods----------------
BaseDataLoader.__init__(self,   )
DataLoader.__init__(self, path, is_training,  vocab=None, params=None,  )
DataLoader.build_vocab(self, data,   )
DataLoader.load_data(self, path,   )
DataLoader.padding(self, data, lens,   )
QA.__init__(self, vocab_size,   )

---------------functions---------------
bAbI_data_load(path,  END=["<end>"],  )
embed_seq(x, vocab_size,  zero_pad=True,  )
hop_forward(question, memory_o, memory_i, response_proj, inputs_len, questions_len, is_training,   )
input_mem(x, vocab_size, max_sent_len, is_training,   )
position_encoding(sentence_size, embedding_size,   )
post_softmax_masking(x, seq_len,   )
pre_softmax_masking(x, seq_len,   )
quest_mem(x, vocab_size, max_quest_len, is_training,   )
shift_right(x,   )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\2.birnn-ctc-greedy.py
----------------methods----------------
Model.__init__(self, num_layers, size_layers, learning_rate, num_features,  dropout=1.0,  )

---------------functions---------------
sparse_tuple_from(sequences,  dtype=np.int32,  )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\3.birnn-ctc-beam.py
----------------methods----------------
Model.__init__(self, num_layers, size_layers, learning_rate, num_features,  dropout=1.0,  )

---------------functions---------------
sparse_tuple_from(sequences,  dtype=np.int32,  )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\4.seq2seq-bahdanau-ctc-beam.py
----------------methods----------------
Model.__init__(self, num_layers, size_layers, learning_rate, num_features,  dropout=1.0,  )

---------------functions---------------
clip_grads(loss_op,   )
sparse_tuple_from(sequences,  dtype=np.int32,  )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\5.seq2seq-luong-ctc-beam.py
----------------methods----------------
Model.__init__(self, num_layers, size_layers, learning_rate, num_features,  dropout=1.0,  )

---------------functions---------------
clip_grads(loss_op,   )
sparse_tuple_from(sequences,  dtype=np.int32,  )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\6.birnn-attention-ctc-beam.py
----------------methods----------------
Model.__init__(self, num_layers, size_layers, learning_rate, num_features,  dropout=1.0,  )

---------------functions---------------
attention(inputs, attention_size,   )
sparse_tuple_from(sequences,  dtype=np.int32,  )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\download.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\speech-to-text\1.tacotron\caching.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\speech-to-text\1.tacotron\model.py
----------------methods----------------
Model.__init__(self,  is_training=True,  )

---------------functions---------------
decode(inputs, memory,  is_training=True, scope="decoder_layers", reuse=None,  )
encode(inputs,  is_training=True, scope="encoder", reuse=None,  )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\1.tacotron\modules.py
----------------methods----------------

---------------functions---------------
attention_decoder(inputs, memory,  units=None, scope="attention_decoder", reuse=None,  )
conv1d(inputs,  filters=None, size=1, rate=1, padding="SAME", use_bias=False, activation_fn=None, scope="conv1d", reuse=None,  )
conv1d_banks(inputs,  K=16, is_training=True, scope="conv1d_banks", reuse=None,  )
embed(inputs, vocab_size, dimension,  scope="embedding", reuse=None,  )
gru(inputs,  units=None, bidirection=False, scope="gru", reuse=None,  )
highwaynet(inputs,  units=None, scope="highwaynet", reuse=None,  )
normalize_bn(inputs,  decay=0.99, is_training=True, activation_fn=None, scope="normalize_bn",  )
normalize_in(inputs,  activation_fn=None, scope="normalize_in",  )
normalize_layer_norm(inputs,  activation_fn=None, scope="normalize_layer_norm",  )
prenet(inputs,  is_training=True, scope="prenet", reuse=None,  )
shift_by_one(inputs,   )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\1.tacotron\setting.py
----------------methods----------------

---------------functions---------------
get_cached(path,   )
get_spectrogram(fpath,   )
load_file(path,   )
reduce_frames(x, r_factor,   )
restore_shape(x, r_factor,   )
text2idx(text,   )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\1.tacotron\train.py
----------------methods----------------

---------------functions---------------
dynamic_batching(paths,   )


mlmodels\model_tf\raw\tf_nlp\speech-to-text\7.wavenet\wavenet.py
----------------methods----------------
Model.__init__(self, num_layers, size_layers, learning_rate, num_features,  num_blocks=3, block_size=128, dropout=1.0,  )

---------------functions---------------
pad_causal(x, size, rate,   )
sparse_tuple_from(sequences,  dtype=np.int32,  )


mlmodels\model_tf\raw\tf_nlp\stemming\1.lstm-seq2seq-beam.py
----------------methods----------------
Stemmer.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
check_accuracy(logits, Y,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\stemming\2.gru-seq2seq-beam.py
----------------methods----------------
Stemmer.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
check_accuracy(logits, Y,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\stemming\3.lstm-birnn-seq2seq-greedy.py
----------------methods----------------
Stemmer.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
check_accuracy(logits, Y,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\stemming\4.gru-birnn-seq2seq-greedy.py
----------------methods----------------
Stemmer.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
check_accuracy(logits, Y,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\stemming\5.dnc-seq2seq-bahdanau-greedy.py
----------------methods----------------
Stemmer.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  attn_input_feeding=True,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
check_accuracy(logits, Y,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\stemming\access.py
----------------methods----------------
MemoryAccess.__init__(self,  memory_size=128, word_size=20, num_reads=1, num_writes=1, name="memory_access",  )
MemoryAccess._build(self, inputs, prev_state,   )
MemoryAccess._read_inputs(self, inputs,   )
MemoryAccess._read_weights(self, inputs, memory, prev_read_weights, link,   )
MemoryAccess._write_weights(self, inputs, memory, usage,   )
MemoryAccess.output_size(self,   )
MemoryAccess.state_size(self,   )

---------------functions---------------
_erase_and_write(memory, address, reset_weights, values,   )


mlmodels\model_tf\raw\tf_nlp\stemming\addressing.py
----------------methods----------------
CosineWeights.__init__(self, num_heads, word_size,  strength_op=tf.nn.softplus, name="cosine_weights",  )
CosineWeights._build(self, memory, keys, strengths,   )
Freeness.__init__(self, memory_size,  name="freeness",  )
Freeness._allocation(self, usage,   )
Freeness._build(self, write_weights, free_gate, read_weights, prev_usage,   )
Freeness._usage_after_read(self, prev_usage, free_gate, read_weights,   )
Freeness._usage_after_write(self, prev_usage, write_weights,   )
Freeness.state_size(self,   )
Freeness.write_allocation_weights(self, usage, write_gates, num_writes,   )
TemporalLinkage.__init__(self, memory_size, num_writes,  name="temporal_linkage",  )
TemporalLinkage._build(self, write_weights, prev_state,   )
TemporalLinkage._link(self, prev_link, prev_precedence_weights, write_weights,   )
TemporalLinkage._precedence_weights(self, prev_precedence_weights, write_weights,   )
TemporalLinkage.directional_read_weights(self, link, prev_read_weights, forward,   )
TemporalLinkage.state_size(self,   )

---------------functions---------------
_vector_norms(m,   )
weighted_softmax(activations, strengths, strengths_op,   )


mlmodels\model_tf\raw\tf_nlp\stemming\dnc-seq2seq-bahdanau-greedy.py
----------------methods----------------
Stemmer.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, learning_rate, batch_size,  attn_input_feeding=True,  )

---------------functions---------------
build_dataset(words, n_words,  atleast=1,  )
check_accuracy(logits, Y,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,   )


mlmodels\model_tf\raw\tf_nlp\stemming\dnc.py
----------------methods----------------
DNC.__init__(self, access_config, controller_config, output_size,  clip_value=None, name="dnc",  )
DNC._build(self, inputs, prev_state,   )
DNC._clip_if_enabled(self, x,   )
DNC.initial_state(self, batch_size,  dtype=tf.float32,  )
DNC.output_size(self,   )
DNC.state_size(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\stemming\util.py
----------------methods----------------

---------------functions---------------
batch_gather(values, indices,   )
batch_invert_permutation(permutations,   )
one_hot(length, index,   )


mlmodels\model_tf\raw\tf_nlp\summarization\1.lstm-seq2seq-greedy.py
----------------methods----------------
Summarization.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size,   )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,  UNK=3,  )
topic_modelling(string,  n=500,  )


mlmodels\model_tf\raw\tf_nlp\summarization\2.lstm-seq2seq-greedy-luong.py
----------------methods----------------
Summarization.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,  UNK=3,  )
topic_modelling(string,  n=500,  )


mlmodels\model_tf\raw\tf_nlp\summarization\3.lstm-seq2seq-beam.py
----------------methods----------------
Summarization.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  dropout=0.5, beam_width=15,  )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,  UNK=3,  )
topic_modelling(string,  n=500,  )


mlmodels\model_tf\raw\tf_nlp\summarization\4.lstm-birnn-seq2seq-beam-luong.py
----------------methods----------------
Summarization.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,  grad_clip=5.0, beam_width=5, force_teaching_ratio=0.5,  )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,  UNK=3,  )
topic_modelling(string,  n=500,  )


mlmodels\model_tf\raw\tf_nlp\summarization\5.lstm-seq2seq-greedy-luong-pointer-generator.py
----------------methods----------------
Summarization.__init__(self, size_layer, num_layers, embedded_size, from_dict_size, to_dict_size, batch_size,   )

---------------functions---------------
build_dataset(words, n_words,   )
pad_sentence_batch(sentence_batch, pad_int,   )
str_idx(corpus, dic,  UNK=3,  )
textcleaning(string,   )
topic_modelling(string,  n=500,  )


mlmodels\model_tf\raw\tf_nlp\summarization\6.bytenet.py
----------------methods----------------
ByteNet.__init__(self, from_vocab_size, to_vocab_size, channels, encoder_dilations, decoder_dilations, encoder_filter_width, decoder_filter_width,  learning_rate=0.001, beta1=0.5,  )

---------------functions---------------
build_dataset(words, n_words,   )
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width,  causal=True,  )
conv1d(input_, output_channels,  dilation=1, filter_width=1, causal=False,  )
layer_normalization(x,  epsilon=1e-8,  )
pad_sentence_batch(sentence_batch, pad_int, maxlen,   )
str_idx(corpus, dic,  UNK=3,  )
topic_modelling(string,  n=500,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\1.basic-rnn.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\10.lstm-rnn-bidirectional.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\11.lstm-rnn-bidirectional-huber.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\12.lstm-rnn-dropout-l2.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\13.gru-rnn.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\14.gru-rnn-hinge.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\15.gru-rnn-huber.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\16.gru-rnn-bidirectional.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\17.gru-rnn-bidirectional-hinge.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\18.gru-rnn-bidirectional-huber.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\19.lstm-cnn-rnn.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate, filter_sizes, pooling_size,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\2.basic-rnn-hinge.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\20.kmax-cnn.py
----------------methods----------------
Model.__init__(self, embedded_size, dict_size, dimension_output, learning_rate,  top_k=5, n_filters=250,  )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\21.lstm-cnn-rnn-highway.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, maxlen,  grad_clip=5.0, kernel_sizes=[3, 3, 3],  )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\22.lstm-rnn-attention.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,  attention_size=150,  )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\23.dilated-rnn-lstm.py
----------------methods----------------
Model.__init__(self, steps, dict_size, dimension_input, dimension_output,  learning_rate=1e-2, hidden_structs=[20], dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],  )

---------------functions---------------
contruct_cells(hidden_structs,   )
dilated_rnn(cell, inputs, rate,  scope="default",  )
multi_dilated_rnn(cells, inputs, dilations,   )
rnn_reformat(x, input_dims, n_steps,   )


mlmodels\model_tf\raw\tf_nlp\text-classification\24.lnlstm-rnn.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\25.only-attention.py
----------------methods----------------
Model.__init__(self, seq_len, dict_size, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------
sinusoidal_positional_encoding(inputs, num_units,  zero_pad=False, scale=False,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\26.multihead-attention.py
----------------methods----------------
Model.__init__(self, dict_size, dimension_input, dimension_output, seq_len, learning_rate,  num_heads=8, attn_windows=range(1, 6),  )
Model.multihead_attn(self, inputs, masks,   )
Model.window_mask(self, h_w,   )

---------------functions---------------
embed_seq(inputs,  vocab_size=None, embed_dim=None, zero_pad=False, scale=False,  )
layer_norm(inputs,  epsilon=1e-8,  )
learned_positional_encoding(inputs, embed_dim,  zero_pad=False, scale=False,  )
pointwise_feedforward(inputs,  num_units=[None, None], activation=None,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\27.neural-turing-machine.py
----------------methods----------------
Model.__init__(self, seq_len, size_layer, batch_size, dict_size, dimension_input, dimension_output, learning_rate, memory_size, memory_vector_size,  read_head_num=4, write_head_num=1,  )
NTMCell.__call__(self, x, prev_state,   )
NTMCell.__init__(self, rnn_size, memory_size, memory_vector_dim, read_head_num, write_head_num,  addressing_mode="content_and_location", shift_range=1, reuse=False, output_dim=None,  )
NTMCell.addressing(self, k, beta, g, s, gamma, prev_M, prev_w,   )
NTMCell.zero_state(self, batch_size, dtype,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\28.lstm-seq2seq.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\29.lstm-seq2seq-luong.py
----------------methods----------------
Model.__init__(self, batch_size, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\3.basic-rnn-huber.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\30.lstm-seq2seq-bahdanau.py
----------------methods----------------
Model.__init__(self, batch_size, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\31.lstm-seq2seq-beam.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\32.lstm-seq2seq-birnn.py
----------------methods----------------
Model.__init__(self, batch_size, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\33.pointer-net.py
----------------methods----------------
Model.__init__(self, size_layer, embedded_size, maxlen, batch_size, dict_size, dimension_output,  grad_clip=5.0,  )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\34.lstm-rnn-bahdanau.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\35.lstm-rnn-luong.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\36.lstm-rnn-bahdanau-luong.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\37.lstm-birnn-bahdanau-luong.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\38.bytenet.py
----------------methods----------------
ByteNet.__init__(self, dict_size, channels, encoder_dilations, dimension_output, encoder_filter_width,  learning_rate=0.001, beta1=0.5,  )

---------------functions---------------
bytenet_residual_block(input_, dilation, layer_no, residual_channels, filter_width,  causal=True,  )
conv1d(input_, output_channels,  dilation=1, filter_width=1, causal=False,  )
layer_normalization(x,  epsilon=1e-8,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\39.fast-slow-lstm.py
----------------methods----------------
FSRNNCell.__call__(self, inputs, state,  scope="FS-RNN",  )
FSRNNCell.__init__(self, fast_cells, slow_cell,  keep_prob=1.0, training=True,  )
FSRNNCell.zero_state(self, batch_size, dtype,   )
LN_LSTMCell.__call__(self, x, state,  scope=None,  )
LN_LSTMCell.__init__(self, num_units,  f_bias=1.0, use_zoneout=False, zoneout_keep_h=0.9, zoneout_keep_c=0.5, is_training=False,  )
LN_LSTMCell.zero_state(self, batch_size, dtype,   )
Model.__init__(self, size_layer, num_layers, fast_layers, embedded_size, dict_size, dimension_output, learning_rate, batch_size, timestamp,  is_training=True, zoneout_h=0.95, zoneout_c=0.7, keep_prob=0.75,  )

---------------functions---------------
layer_norm(x,  scope="layer_norm", alpha_start=1.0, bias_start=0.0,  )
layer_norm_all(h, base, num_units, scope,   )
moments_for_layer_norm(x,  axes=1, name=None,  )
zoneout(new_h, new_c, h, c, h_keep, c_keep, is_training,   )


mlmodels\model_tf\raw\tf_nlp\text-classification\4.basic-rnn-bidirectional.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\40.siamese-network.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output,  margin=0.2,  )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\41.estimator.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, batch_size, from_dict_size, to_dict_size,  grad_clip=5.0,  )
Model.lstm_cell(self,  reuse=False,  )
Model.model_fn(self, features, labels, mode,   )
Model.seq2seq(self, x_dict,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\42.capsule-rnn-lstm.py
----------------methods----------------
CapsuleNetwork.__init__(self, batch_size, learning_rate, seq_len, size_layer, num_layers, maxlen, dict_size, embedded_size, dimension_output,  kernels=[6, 3, 2], strides=[3, 2, 1], epsilon=1e-8,  )

---------------functions---------------
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\43.capsule-seq2seq-lstm.py
----------------methods----------------
CapsuleNetwork.__init__(self, batch_size, learning_rate, seq_len, size_layer, num_layers, maxlen, dict_size, embedded_size, dimension_output,  kernels=[6, 3, 2], strides=[3, 2, 1], epsilon=1e-8, skip=5,  )

---------------functions---------------
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\44.capsule-birrn-seq2seq-lstm.py
----------------methods----------------
CapsuleNetwork.__init__(self, batch_size, learning_rate, seq_len, size_layer, num_layers, maxlen, dict_size, embedded_size, dimension_output,  kernels=[6, 3, 2], strides=[3, 2, 1], epsilon=1e-8, skip=5,  )

---------------functions---------------
conv_layer(X, num_output, num_vector,  kernel=None, stride=None,  )
fully_conn_layer(X, num_output, dimension_out,   )
routing(X, b_IJ, seq_len, dimension_out,  routing_times=2,  )
squash(X,  epsilon=1e-9,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\45.nested-lstm.py
----------------methods----------------
Model.__init__(self, size_layer, embedded_size, dict_size, dimension_output, learning_rate, batch_size, timestamp,  depth=1,  )
NLSTMCell.__init__(self, num_units, depth,  forget_bias=1.0, state_is_tuple=True, use_peepholes=True, activation=None, gate_activation=None, cell_activation=None, initializer=None, input_gate_initializer=None, use_bias=True, reuse=None, name=None,  )
NLSTMCell._recurrence(self, inputs, hidden_state, cell_states, depth,   )
NLSTMCell.build(self, inputs_shape,   )
NLSTMCell.call(self, inputs, state,   )
NLSTMCell.depth(self,   )
NLSTMCell.output_size(self,   )
NLSTMCell.state_size(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\46.lstm-seq2seq-highway.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, maxlen, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\47.triplet-loss-lstm.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output,  margin=0.2,  )

---------------functions---------------
compute_euclidean_distance(x, y,   )
compute_triplet_loss(anchor_feature, positive_feature, negative_feature,  margin=0.01,  )
get_one_triplet(input_data, input_labels, n_labels,   )


mlmodels\model_tf\raw\tf_nlp\text-classification\48.dnc.py
----------------methods----------------
Model.__init__(self, dict_size,   )

---------------functions---------------
run_model(input_sequence, output_size,   )


mlmodels\model_tf\raw\tf_nlp\text-classification\49.convlstm.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,  maxlen=50, conv_len=10,  )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\5.basic-rnn-bidirectional-hinge.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\50.temporalconvd.py
----------------methods----------------
Model.__init__(self, embedded_size, dict_size, dimension_output, learning_rate,  levels=5, size_layer=256, kernel_size=7, maxlen=50,  )

---------------functions---------------
attention_block(x,   )
convolution1d(x, num_filters, dilation_rate, k,  filter_size=3, stride=[1], pad="VALID",  )
temporal_convd(input_layer, num_channels, sequence_length,  kernel_size=2, dropout=0,  )
temporal_padding(x,  padding=(1, 1),  )
temporalblock(input_layer, out_channels, filter_size, stride, dilation_rate, dropout, k,  highway=False,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\51.batch-all-triplet-loss-lstm.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output,   )

---------------functions---------------
_get_anchor_negative_triplet_mask(labels,   )
_get_anchor_positive_triplet_mask(labels,   )
_get_triplet_mask(labels,   )
_pairwise_distances(embeddings,  squared=False,  )
batch_all_triplet_loss(labels, embeddings, margin,  squared=False,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\52.fast-text.py
----------------methods----------------
Model.__init__(self, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------
add_ngram(sequences, token_indice,   )
build_ngram(x_train,   )
create_ngram_set(input_list, ngram_value,   )


mlmodels\model_tf\raw\tf_nlp\text-classification\53.gated-convolution-network.py
----------------methods----------------
Model.__init__(self, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------
gated_linear_unit(x, d_rate,   )


mlmodels\model_tf\raw\tf_nlp\text-classification\54.simple-recurrent-units.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output,   )
SRUCell.__call__(self, inputs, state,  scope="SRUCell",  )
SRUCell.__init__(self, num_units,  activation=None, reuse=None,  )
SRUCell.output_size(self,   )
SRUCell.state_size(self,   )

---------------functions---------------
linear(args, output_size, bias,  bias_start=0.0, scope=None,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\55.lstm-han.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\56.bert.py
----------------methods----------------

---------------functions---------------
create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels, use_one_hot_embeddings,  reuse_flag=False,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\57.dynamic-memory-network.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\58.entity-network.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\59.memory-network.py
----------------methods----------------
QA.__init__(self, vocab_size, size_layer, learning_rate, dimension_output,  n_hops=3,  )

---------------functions---------------
embed_seq(x, vocab_size,  zero_pad=True,  )
hop_forward(memory_o, memory_i, response_proj, inputs_len, questions_len,   )
position_encoding(sentence_size, embedding_size,   )
post_softmax_masking(x, seq_len,   )
pre_softmax_masking(x, seq_len,   )
quest_mem(x, vocab_size, max_quest_len,   )
shift_right(x,   )


mlmodels\model_tf\raw\tf_nlp\text-classification\6.basic-rnn-bidirectional-huber.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\60.char-sparse.py
----------------methods----------------
Model.__init__(self, output_size, vocab_size, learning_rate,   )

---------------functions---------------
convert_sparse_matrix_to_sparse_tensor(X,  limit=5,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\61.residual-network.py
----------------methods----------------
Model.__init__(self, dict_size, size_layers, learning_rate, num_classes,  num_blocks=3, block_size=128,  )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\62.residual-network-bahdanau.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Model.__init__(self, dict_size, size_layers, learning_rate, num_classes, maxlen,  num_blocks=3, block_size=128,  )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\63.deep-pyramid-cnn.py
----------------methods----------------
Model.__init__(self, maxlen, dimension_output, vocab_size, embedding_size, kernel_size, num_filters, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\64.transformer-xl.py
----------------methods----------------
Model.__init__(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\65.gpt-2.py
----------------methods----------------
Model.__init__(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\66.quasi-rnn.py
----------------methods----------------
Model.__init__(self,   )
QRNN_layer.__call__(self, input_,   )
QRNN_layer.__init__(self, out_fmaps,  fwidth=2, activation=tf.tanh, pool_type="fo", zoneout=0.1, infer=False, bias_init_val=None, name="QRNN",  )
QRNN_layer.convolution(self, input_, filter_width, out_fmaps, zoneout_,   )
QRNN_pooling.__call__(self, inputs, state,  scope=None,  )
QRNN_pooling.__init__(self, out_fmaps,   )
QRNN_pooling.output_size(self,   )
QRNN_pooling.state_size(self,   )

---------------functions---------------
zoneout(x, keep_prob,  noise_shape=None, seed=None, name=None,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\7.lstm-rnn.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\8.lstm-rnn-hinge.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\9.lstm-rnn-huber.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\bert_model.py
----------------methods----------------
BertConfig.__init__(self, vocab_size,  hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02,  )
BertConfig.from_dict(cls, json_object,   )
BertConfig.from_json_file(cls, json_file,   )
BertConfig.to_dict(self,   )
BertConfig.to_json_string(self,   )
BertModel.__init__(self, config, is_training, input_ids,  input_mask=None, token_type_ids=None, use_one_hot_embeddings=True, scope=None,  )
BertModel.get_all_encoder_layers(self,   )
BertModel.get_embedding_output(self,   )
BertModel.get_embedding_table(self,   )
BertModel.get_pooled_output(self,   )
BertModel.get_sequence_output(self,   )
BertModel.transpose_for_scores(input_tensor, batch_size, num_attention_heads, seq_length, width,   )

---------------functions---------------
assert_rank(tensor, expected_rank,  name=None,  )
attention_layer(from_tensor, to_tensor,  attention_mask=None, num_attention_heads=1, size_per_head=512, query_act=None, key_act=None, value_act=None, attention_probs_dropout_prob=0.0, initializer_range=0.02, do_return_2d_tensor=False, batch_size=None, from_seq_length=None, to_seq_length=None,  )
create_attention_mask_from_input_mask(from_tensor, to_mask,   )
create_initializer( initializer_range=0.02,  )
dropout(input_tensor, dropout_prob,   )
embedding_lookup(input_ids, vocab_size,  embedding_size=128, initializer_range=0.02, word_embedding_name="word_embeddings", use_one_hot_embeddings=False,  )
embedding_postprocessor(input_tensor,  use_token_type=False, token_type_ids=None, token_type_vocab_size=16, token_type_embedding_name="token_type_embeddings", use_position_embeddings=True, position_embedding_name="position_embeddings", initializer_range=0.02, max_position_embeddings=512, dropout_prob=0.1,  )
gelu(input_tensor,   )
get_activation(activation_string,   )
get_assignment_map_from_checkpoint(tvars, init_checkpoint,   )
get_shape_list(tensor,  expected_rank=None, name=None,  )
layer_norm(input_tensor,  name=None,  )
layer_norm_and_dropout(input_tensor, dropout_prob,  name=None,  )
reshape_from_matrix(output_tensor, orig_shape_list,   )
reshape_to_matrix(input_tensor,   )
transformer_model(input_tensor,  attention_mask=None, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, intermediate_act_fn=gelu, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, initializer_range=0.02, do_return_all_layers=False,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\dynamic_memory_network.py
----------------methods----------------
DynamicMemoryNetwork.__init__(self, num_classes, learning_rate, decay_steps, decay_rate, sequence_length, story_length, vocab_size, embed_size, hidden_size,  num_pass=2, use_gated_gru=True, decode_with_sequences=False, initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0, l2_lambda=0.0001,  )
DynamicMemoryNetwork.answer_module(self,   )
DynamicMemoryNetwork.attention_mechanism_parallel(self, c_full, m, q, i,   )
DynamicMemoryNetwork.episodic_memory_module(self,   )
DynamicMemoryNetwork.gated_gru(self, c_current, h_previous, g_current,   )
DynamicMemoryNetwork.gru_cell(self, Xt, h_t_minus_1, variable_scope,   )
DynamicMemoryNetwork.inference(self,   )
DynamicMemoryNetwork.input_module(self,   )
DynamicMemoryNetwork.instantiate_weights(self,   )
DynamicMemoryNetwork.loss(self,  l2_lambda=0.0001,  )
DynamicMemoryNetwork.question_module(self,   )
DynamicMemoryNetwork.train(self,   )
DynamicMemoryNetwork.x1Wx2_parallel(self, x1, x2, scope,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\entity_network.py
----------------methods----------------
EntityNetwork.__init__(self, num_classes, learning_rate, decay_steps, decay_rate, sequence_length, story_length, vocab_size, embed_size, hidden_size,  block_size=20, initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0, use_bi_lstm=False,  )
EntityNetwork.activation(self, features,  scope=None,  )
EntityNetwork.cell(self, s_t, h_all, w_all, i,   )
EntityNetwork.embedding_with_mask(self,   )
EntityNetwork.inference(self,   )
EntityNetwork.input_encoder_bi_lstm(self,   )
EntityNetwork.input_encoder_bow(self,   )
EntityNetwork.instantiate_weights(self,   )
EntityNetwork.loss(self,  l2_lambda=0.0001,  )
EntityNetwork.output_module(self,   )
EntityNetwork.rnn_story(self,   )
EntityNetwork.train(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-classification\gpt_2.py
----------------methods----------------

---------------functions---------------
attention_mask(nd, ns,   )
attn(x, scope, n_state,   )
block(x, scope,   )
conv1d(x, scope, nf,   )
expand_tile(value, size,   )
gelu(x,   )
merge_states(x,   )
mlp(x, scope, n_state,   )
model(hparams, X,  past=None, scope="model", reuse=False,  )
norm(x, scope,   )
past_shape(  )
positions_for(tokens, past_length,   )
shape_list(x,   )
softmax(x,  axis=-1,  )
split_states(x, n,   )


mlmodels\model_tf\raw\tf_nlp\text-classification\utils.py
----------------methods----------------

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
separate_dataset(trainset,  ratio=0.5,  )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_nlp\text-classification\xl.py
----------------methods----------------

---------------functions---------------
_cache_mem(curr_out, prev_mem,  mem_len=None,  )
_create_mask(qlen, mlen,  same_length=False,  )
embedding_lookup(lookup_table, x,   )
mask_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, proj_same_dim=True, scope="adaptive_embed",  **kwargs)
mask_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs, params, tie_projs,  initializer=None, proj_initializer=None, div_val=1, scope="adaptive_softmax", proj_same_dim=True, return_mean=True,  **kwargs)
mul_adaptive_embedding_lookup(x, n_token, d_embed, d_proj, cutoffs, initializer, proj_initializer,  div_val=1, perms=None, proj_same_dim=True, scope="adaptive_embed",  )
mul_adaptive_logsoftmax(hidden, target, n_token, d_embed, d_proj, cutoffs, params, tie_projs,  initializer=None, proj_initializer=None, div_val=1, perms=None, proj_same_dim=True, scope="adaptive_softmax",  **kwargs)
positional_embedding(pos_seq, inv_freq,  bsz=None,  )
positionwise_FF(inp, d_model, d_inner, kernel_initializer,  scope="ff",  )
rel_multihead_attn(w, r, r_w_bias, r_r_bias, attn_mask, mems, d_model, n_head, d_head, kernel_initializer,  scope="rel_attn",  )
rel_shift(x,   )
transformer(dec_inp, mems, n_token, n_layer, d_model, d_embed, n_head, d_head, d_inner, initializer,  proj_initializer=None, mem_len=None, cutoffs=[], div_val=1, tie_projs=[], same_length=False, clamp_len=-1, untie_r=False, proj_same_dim=True, scope="transformer",  )


mlmodels\model_tf\raw\tf_nlp\text-similarity\1.char-similarity-siamese-birnn.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, dropout,   )

---------------functions---------------
build_dataset(words, n_words,   )
load_data(filepath,   )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_nlp\text-similarity\2.sentence-similarity-birnn.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, dropout,   )

---------------functions---------------
build_dataset(words, n_words,   )
load_data(filepath,   )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_nlp\text-similarity\3.char-similarity-batchall-tripletloss.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, dimension_output,   )

---------------functions---------------
_get_anchor_negative_triplet_mask(labels,   )
_get_anchor_positive_triplet_mask(labels,   )
_get_triplet_mask(labels,   )
_pairwise_distances(embeddings_left, embeddings_right,  squared=False,  )
batch_all_triplet_loss(labels, embeddings_left, embeddings_right, margin,  squared=False,  )
build_dataset(words, n_words,   )
load_data(filepath,   )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_nlp\text-similarity\4.sentence-similarity-batchall-tripletloss.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, learning_rate, dimension_output,   )

---------------functions---------------
_get_anchor_negative_triplet_mask(labels,   )
_get_anchor_positive_triplet_mask(labels,   )
_get_triplet_mask(labels,   )
_pairwise_distances(embeddings_left, embeddings_right,  squared=False,  )
batch_all_triplet_loss(labels, embeddings_left, embeddings_right, margin,  squared=False,  )
build_dataset(words, n_words,   )
load_data(filepath,   )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_nlp\text-to-speech\3.seq2seq-luong.py
----------------methods----------------
Model.__init__(self, num_layers, size_layers,  learning_rate=1e-3, dropout=1.0,  )

---------------functions---------------
conv1d_banks(inputs,  K=16, is_training=True, scope="conv1d_banks",  )
dynamic_batching(paths,   )
highwaynet(inputs,  num_units=None, scope="highwaynet",  )
prenet(inputs,  num_units=None, is_training=True, scope="prenet",  )


mlmodels\model_tf\raw\tf_nlp\text-to-speech\4.seq2seq-bahdanau.py
----------------methods----------------
Model.__init__(self, num_layers, size_layers,  learning_rate=1e-3, dropout=1.0,  )

---------------functions---------------
conv1d_banks(inputs,  K=16, is_training=True, scope="conv1d_banks",  )
dynamic_batching(paths,   )
highwaynet(inputs,  num_units=None, scope="highwaynet",  )
prenet(inputs,  num_units=None, is_training=True, scope="prenet",  )


mlmodels\model_tf\raw\tf_nlp\text-to-speech\caching.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-to-speech\download.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-to-speech\utils.py
----------------methods----------------

---------------functions---------------
get_cached(path,   )
get_spectrogram(audio_file,   )
get_wav(spectrogram,   )
griffin_lim(spectrogram,   )
invert_spectrogram(spectrogram,   )
load_file(path,   )
plot_alignment(alignment,   )
spectrogram2wav(mag,   )
text_normalize(text,   )


mlmodels\model_tf\raw\tf_nlp\text-to-speech\1.tacotron\caching.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-to-speech\1.tacotron\tacotron.py
----------------methods----------------
Tacotron.__init__(self,  is_training=True, reuse=None,  )

---------------functions---------------
conv1d_banks(inputs,  K=16, is_training=True, scope="conv1d_banks",  )
highwaynet(inputs,  num_units=None, scope="highwaynet",  )
prenet(inputs,  num_units=None, is_training=True, scope="prenet",  )


mlmodels\model_tf\raw\tf_nlp\text-to-speech\1.tacotron\train.py
----------------methods----------------

---------------functions---------------
dynamic_batching(paths,   )


mlmodels\model_tf\raw\tf_nlp\text-to-speech\1.tacotron\utils.py
----------------methods----------------

---------------functions---------------
get_cached(path,   )
get_spectrogram(audio_file,   )
get_wav(spectrogram,   )
griffin_lim(spectrogram,   )
invert_spectrogram(spectrogram,   )
load_file(path,   )
plot_alignment(alignment,   )
spectrogram2wav(mag,   )
text_normalize(text,   )


mlmodels\model_tf\raw\tf_nlp\text-to-speech\2.wavenet\caching.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_nlp\text-to-speech\2.wavenet\utils.py
----------------methods----------------

---------------functions---------------
get_cached(path,   )
get_spectrogram(audio_file,   )
get_wav(spectrogram,   )
griffin_lim(spectrogram,   )
invert_spectrogram(spectrogram,   )
load_file(path,   )
plot_alignment(alignment,   )
spectrogram2wav(mag,   )
text_normalize(text,   )


mlmodels\model_tf\raw\tf_nlp\text-to-speech\2.wavenet\wavenet.py
----------------methods----------------
Wavenet.__init__(self, num_layers, size_layers,  num_blocks=3, block_size=128, dropout=1.0,  )

---------------functions---------------
dynamic_batching(paths,   )
pad_causal(x, size, rate,   )


mlmodels\model_tf\raw\tf_nlp\unsupervised-summarization\1.skip-thought.py
----------------methods----------------
Model.__init__(self,  maxlen=50, vocabulary_size=20000, learning_rate=1e-3, embedding_size=256,  )
Model.calculate_loss(self, outputs, labels,   )
Model.decoder(self, thought, labels,   )
Model.get_embedding(self, inputs,   )
Model.thought(self, inputs,   )

---------------functions---------------
batch_sequence(sentences, dictionary,  maxlen=50,  )
build_dict(word_counter,  vocab_size=50000,  )
counter_words(sentences,   )
simple_textcleaning(string,   )
split_by_dot(string,   )


mlmodels\model_tf\raw\tf_nlp\unsupervised-summarization\2.residual-network.py
----------------methods----------------
Model.__init__(self, dict_size, size_layers, learning_rate, maxlen,  num_blocks=3,  )
Model.calculate_loss(self, outputs, labels,   )

---------------functions---------------
batch_sequence(sentences, dictionary,  maxlen=50,  )
build_dict(word_counter,  vocab_size=50000,  )
counter_words(sentences,   )
simple_textcleaning(string,   )
split_by_dot(string,   )


mlmodels\model_tf\raw\tf_nlp\unsupervised-summarization\3.residual-network-bahdanau.py
----------------methods----------------
Attention.__call__(self, hidden, encoder_outputs,   )
Attention.__init__(self, hidden_size,   )
Attention.score(self, hidden_tensor, encoder_outputs,   )
Model.__init__(self, dict_size, size_layers, learning_rate, maxlen,  num_blocks=3,  )
Model.calculate_loss(self, outputs, labels,   )

---------------functions---------------
batch_sequence(sentences, dictionary,  maxlen=50,  )
build_dict(word_counter,  vocab_size=50000,  )
counter_words(sentences,   )
simple_textcleaning(string,   )
split_by_dot(string,   )


mlmodels\model_tf\raw\tf_serving\1.object-detection-flasksocketio-webrtc\app.py
----------------methods----------------

---------------functions---------------
gen_livestream(  )
home(  )
test_connect(  )
test_disconnect(  )
test_live(message,   )
test_message(message,   )
video_feed(  )


mlmodels\model_tf\raw\tf_serving\1.object-detection-flasksocketio-webrtc\object_detection.py
----------------methods----------------

---------------functions---------------
detect_object(img,   )


mlmodels\model_tf\raw\tf_serving\1.object-detection-flasksocketio-webrtc\utils\label_map_util.py
----------------methods----------------

---------------functions---------------
_validate_label_map(label_map,   )
convert_label_map_to_categories(label_map, max_num_classes,  use_display_name=True,  )
create_category_index(categories,   )
create_category_index_from_labelmap(label_map_path,   )
create_class_agnostic_category_index(  )
get_label_map_dict(label_map_path,  use_display_name=False,  )
get_max_label_map_index(label_map,   )
load_labelmap(path,   )


mlmodels\model_tf\raw\tf_serving\1.object-detection-flasksocketio-webrtc\utils\standard_fields.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\1.object-detection-flasksocketio-webrtc\utils\string_int_label_map_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\1.object-detection-flasksocketio-webrtc\utils\visualization_utils.py
----------------methods----------------

---------------functions---------------
_visualize_boxes(image, boxes, classes, scores, category_index,   **kwargs)
_visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints, category_index,   **kwargs)
_visualize_boxes_and_masks(image, boxes, classes, scores, masks, category_index,   **kwargs)
_visualize_boxes_and_masks_and_keypoints(image, boxes, classes, scores, masks, keypoints, category_index,   **kwargs)
add_cdf_image_summary(values, name,   )
add_hist_image_summary(values, bins, name,   )
draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_boxes_on_image(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_array(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_tensors(images, boxes, classes, scores, category_index,  instance_masks=None, keypoints=None, max_boxes_to_draw=20, min_score_thresh=0.2,  )
draw_keypoints_on_image(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_keypoints_on_image_array(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_mask_on_image_array(image, mask,  color="red", alpha=0.4,  )
draw_side_by_side_evaluation_image(eval_dict, category_index,  max_boxes_to_draw=20, min_score_thresh=0.2,  )
encode_image_array_as_png_str(image,   )
save_image_array_as_png(image, output_path,   )
visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,  instance_masks=None, instance_boundaries=None, keypoints=None, use_normalized_coordinates=False, max_boxes_to_draw=20, min_score_thresh=0.5, agnostic_mode=False, line_thickness=4, groundtruth_box_visualization_color="black", skip_scores=False, skip_labels=False,  )


mlmodels\model_tf\raw\tf_serving\10.inception-docker\app.py
----------------methods----------------

---------------functions---------------
inception(  )


mlmodels\model_tf\raw\tf_serving\10.inception-docker\inception_utils.py
----------------methods----------------

---------------functions---------------
inception_arg_scope( weight_decay=0.00004, use_batch_norm=True, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, activation_fn=tf.nn.relu, batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,  )


mlmodels\model_tf\raw\tf_serving\10.inception-docker\inception_v1.py
----------------methods----------------

---------------functions---------------
inception_v1(inputs,  num_classes=1000, is_training=True, dropout_keep_prob=0.8, prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None, scope="InceptionV1", global_pool=False,  )
inception_v1_base(inputs,  final_endpoint="Mixed_5c", scope="InceptionV1",  )


mlmodels\model_tf\raw\tf_serving\11.inception-docker-swarm-nginx\app.py
----------------methods----------------

---------------functions---------------
inception(  )


mlmodels\model_tf\raw\tf_serving\11.inception-docker-swarm-nginx\inception_utils.py
----------------methods----------------

---------------functions---------------
inception_arg_scope( weight_decay=0.00004, use_batch_norm=True, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, activation_fn=tf.nn.relu, batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,  )


mlmodels\model_tf\raw\tf_serving\11.inception-docker-swarm-nginx\inception_v1.py
----------------methods----------------

---------------functions---------------
inception_v1(inputs,  num_classes=1000, is_training=True, dropout_keep_prob=0.8, prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None, scope="InceptionV1", global_pool=False,  )
inception_v1_base(inputs,  final_endpoint="Mixed_5c", scope="InceptionV1",  )


mlmodels\model_tf\raw\tf_serving\12.text-classification-hadoop\classification.py
----------------methods----------------

---------------functions---------------
load_graph(frozen_graph_filename,   )


mlmodels\model_tf\raw\tf_serving\12.text-classification-hadoop\freeze-model.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
freeze_graph(model_dir, output_node_names,   )
load_graph(frozen_graph_filename,   )
separate_dataset(trainset,  ratio=1,  )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_serving\12.text-classification-hadoop\lowercase.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\12.text-classification-hadoop\mapping.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\12.text-classification-hadoop\reducer.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\13.text-classification-kafka\app.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\13.text-classification-kafka\freeze-model.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
freeze_graph(model_dir, output_node_names,   )
load_graph(frozen_graph_filename,   )
separate_dataset(trainset,  ratio=1,  )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_serving\13.text-classification-kafka\producer.py
----------------methods----------------

---------------functions---------------
connect_kafka_producer(  )
load_graph(frozen_graph_filename,   )
publish_message(producer_instance, topic_name, key, value,   )


mlmodels\model_tf\raw\tf_serving\14.text-classification-distributed-tf\app.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------
hello(  )


mlmodels\model_tf\raw\tf_serving\14.text-classification-distributed-tf\backend.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\14.text-classification-distributed-tf\freeze-model.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
freeze_graph(model_dir, output_node_names,   )
load_graph(frozen_graph_filename,   )
separate_dataset(trainset,  ratio=1,  )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_serving\14.text-classification-distributed-tf\reduce-mean.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\15.text-classification-tornado-gunicorn\app.py
----------------methods----------------
MainHandler.get(self,   )
TextClassification.get(self,   )

---------------functions---------------
load_graph(frozen_graph_filename,   )


mlmodels\model_tf\raw\tf_serving\16.celery-hadoop-flask-text-classification\app.py
----------------methods----------------

---------------functions---------------
classify_text(self,   )
classify_text_status(task_id,   )
get_hadoop_script(input_location, output_location, mapper,  hadoop_location="/opt/hadoop/bin/hadoop", hadoop_streaming="/opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.1.1.jar", files=["dictionary-test.json", "frozen_model.pb"], reducer="reducer.py",  )
id_generator( size=6, chars=string.ascii_uppercase + string.digits,  )
process(  )
upload(  )
upload_files_dfs(self, file_location, split_size,   )
upload_status(task_id,   )


mlmodels\model_tf\raw\tf_serving\16.celery-hadoop-flask-text-classification\classification.py
----------------methods----------------

---------------functions---------------
load_graph(frozen_graph_filename,   )


mlmodels\model_tf\raw\tf_serving\16.celery-hadoop-flask-text-classification\freeze-model.py
----------------methods----------------
Model.__init__(self, size_layer, num_layers, embedded_size, dict_size, dimension_output, learning_rate,   )

---------------functions---------------
build_dataset(words, n_words,   )
clearstring(string,   )
freeze_graph(model_dir, output_node_names,   )
load_graph(frozen_graph_filename,   )
separate_dataset(trainset,  ratio=1,  )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_serving\16.celery-hadoop-flask-text-classification\reducer.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\17.luigi-hadoop-text-classification\textclassification-hadoop.py
----------------methods----------------
Classification_Hadoop.extra_files(self,   )
Classification_Hadoop.mapper(self, line,   )
Classification_Hadoop.output(self,   )
Classification_Hadoop.reducer(self, key, values,   )
Classification_Hadoop.requires(self,   )
Split_text.output(self,   )
Split_text.run(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\17.luigi-hadoop-text-classification\wordcount-hadoop.py
----------------methods----------------
Split_text.output(self,   )
Split_text.run(self,   )
WordCount_Hadoop.mapper(self, line,   )
WordCount_Hadoop.output(self,   )
WordCount_Hadoop.reducer(self, key, values,   )
WordCount_Hadoop.requires(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\17.luigi-hadoop-text-classification\wordcount.py
----------------methods----------------
Split_text.output(self,   )
Split_text.run(self,   )
WordCount.output(self,   )
WordCount.requires(self,   )
WordCount.run(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\18.luigi-celery-text-classification\app.py
----------------methods----------------

---------------functions---------------
hello(  )
luigi_task(self, filename, topic,   )
sentiment(  )
upload(  )
upload_status(task_id,   )


mlmodels\model_tf\raw\tf_serving\18.luigi-celery-text-classification\function.py
----------------methods----------------
Emotion.output(self,   )
Emotion.requires(self,   )
Emotion.run(self,   )
Save_to_Elastic.requires(self,   )
Save_to_Elastic.run(self,   )
Sentiment.output(self,   )
Sentiment.run(self,   )

---------------functions---------------
classify_sentiment(text,   )
clearstring(string,   )
load_graph(frozen_graph_filename,   )
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_serving\19.airflow-elasticsearch-flask\app.py
----------------methods----------------

---------------functions---------------
get_es( index="test_index",  )
get_sentiment(sentiment,   )
hello(  )
indexing(sentiment,   )


mlmodels\model_tf\raw\tf_serving\19.airflow-elasticsearch-flask\dags\sentiment_to_elastic.py
----------------methods----------------

---------------functions---------------
clearstring(string,   )
load_graph(frozen_graph_filename,   )
pull_to_elastic(  **kwargs)
push_sentiment(  **context)
str_idx(corpus, dic, maxlen,  UNK=3,  )


mlmodels\model_tf\raw\tf_serving\19.airflow-elasticsearch-flask\dags\test_print.py
----------------methods----------------

---------------functions---------------
print_hello(  )


mlmodels\model_tf\raw\tf_serving\19.airflow-elasticsearch-flask\dags\test_xcom.py
----------------methods----------------

---------------functions---------------
pull_function(  **kwargs)
push_function(  **context)


mlmodels\model_tf\raw\tf_serving\2.object-detection-flasksocketio-opencv\app.py
----------------methods----------------

---------------functions---------------
test_connect(  )
test_disconnect(  )
test_live(message,   )
test_message(message,   )


mlmodels\model_tf\raw\tf_serving\2.object-detection-flasksocketio-opencv\camera.py
----------------methods----------------

---------------functions---------------
on_camera_response(  *args)
receive_events_thread(  )
run_cam(  )


mlmodels\model_tf\raw\tf_serving\2.object-detection-flasksocketio-opencv\object_detection.py
----------------methods----------------

---------------functions---------------
detect_object(img,   )


mlmodels\model_tf\raw\tf_serving\2.object-detection-flasksocketio-opencv\visual.py
----------------methods----------------

---------------functions---------------
on_camera_response(  *args)
receive_events_thread(  )
run_cam(  )


mlmodels\model_tf\raw\tf_serving\2.object-detection-flasksocketio-opencv\utils\label_map_util.py
----------------methods----------------

---------------functions---------------
_validate_label_map(label_map,   )
convert_label_map_to_categories(label_map, max_num_classes,  use_display_name=True,  )
create_category_index(categories,   )
create_category_index_from_labelmap(label_map_path,   )
create_class_agnostic_category_index(  )
get_label_map_dict(label_map_path,  use_display_name=False,  )
get_max_label_map_index(label_map,   )
load_labelmap(path,   )


mlmodels\model_tf\raw\tf_serving\2.object-detection-flasksocketio-opencv\utils\standard_fields.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\2.object-detection-flasksocketio-opencv\utils\string_int_label_map_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\2.object-detection-flasksocketio-opencv\utils\visualization_utils.py
----------------methods----------------

---------------functions---------------
_visualize_boxes(image, boxes, classes, scores, category_index,   **kwargs)
_visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints, category_index,   **kwargs)
_visualize_boxes_and_masks(image, boxes, classes, scores, masks, category_index,   **kwargs)
_visualize_boxes_and_masks_and_keypoints(image, boxes, classes, scores, masks, keypoints, category_index,   **kwargs)
add_cdf_image_summary(values, name,   )
add_hist_image_summary(values, bins, name,   )
draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_boxes_on_image(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_array(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_tensors(images, boxes, classes, scores, category_index,  instance_masks=None, keypoints=None, max_boxes_to_draw=20, min_score_thresh=0.2,  )
draw_keypoints_on_image(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_keypoints_on_image_array(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_mask_on_image_array(image, mask,  color="red", alpha=0.4,  )
draw_side_by_side_evaluation_image(eval_dict, category_index,  max_boxes_to_draw=20, min_score_thresh=0.2,  )
encode_image_array_as_png_str(image,   )
save_image_array_as_png(image, output_path,   )
visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,  instance_masks=None, instance_boundaries=None, keypoints=None, use_normalized_coordinates=False, max_boxes_to_draw=20, min_score_thresh=0.5, agnostic_mode=False, line_thickness=4, groundtruth_box_visualization_color="black", skip_scores=False, skip_labels=False,  )


mlmodels\model_tf\raw\tf_serving\3.speech-streaming-flasksocketio\client.py
----------------methods----------------

---------------functions---------------
on_speech_response(  *args)
receive_events_thread(  )


mlmodels\model_tf\raw\tf_serving\3.speech-streaming-flasksocketio\server.py
----------------methods----------------

---------------functions---------------
test_connect(  )
test_disconnect(  )
test_live(message,   )
test_message(message,   )


mlmodels\model_tf\raw\tf_serving\4.classification-flask-gunicorn\api_dynamic.py
----------------methods----------------

---------------functions---------------
get_text(  )


mlmodels\model_tf\raw\tf_serving\4.classification-flask-gunicorn\api_static.py
----------------methods----------------

---------------functions---------------
get_text(  )
load_graph(frozen_graph_filename,   )


mlmodels\model_tf\raw\tf_serving\4.classification-flask-gunicorn\comparison-request-gunicorn-dynamic.py
----------------methods----------------

---------------functions---------------
get_time(text, type_api, i,   )
run_parallel_in_threads(target, args_list,   )


mlmodels\model_tf\raw\tf_serving\4.classification-flask-gunicorn\comparison-request.py
----------------methods----------------

---------------functions---------------
get_time(text, type_api, i,   )
run_parallel_in_threads(target, args_list,   )


mlmodels\model_tf\raw\tf_serving\4.classification-flask-gunicorn\model.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\4.classification-flask-gunicorn\sentiment-huber-freeze.py
----------------methods----------------
Model.__init__(self, num_layers, size_layer, dimension_input, dimension_output, learning_rate,   )

---------------functions---------------
freeze_graph(model_dir, output_node_names,   )
load_graph(frozen_graph_filename,   )


mlmodels\model_tf\raw\tf_serving\4.classification-flask-gunicorn\gunicorn\dynamic-comparison.py
----------------methods----------------

---------------functions---------------
convert_to_tuple(gunicorn,   )


mlmodels\model_tf\raw\tf_serving\5.tf-serving\save-serving.py
----------------methods----------------
Model.__init__(self, learning_rate, y_shape,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\classification_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\get_model_metadata_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\inference_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\input_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\model_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\model_service_pb2.py
----------------methods----------------
BetaModelServiceServicer.GetModelStatus(self, request, context,   )
BetaModelServiceStub.GetModelStatus(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
ModelServiceServicer.GetModelStatus(self, request, context,   )
ModelServiceStub.__init__(self, channel,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\model_service_pb2_grpc.py
----------------methods----------------
ModelServiceServicer.GetModelStatus(self, request, context,   )
ModelServiceStub.__init__(self, channel,   )

---------------functions---------------
add_ModelServiceServicer_to_server(servicer, server,   )


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\prediction_service_pb2.py
----------------methods----------------
BetaPredictionServiceServicer.Classify(self, request, context,   )
BetaPredictionServiceServicer.GetModelMetadata(self, request, context,   )
BetaPredictionServiceServicer.MultiInference(self, request, context,   )
BetaPredictionServiceServicer.Predict(self, request, context,   )
BetaPredictionServiceServicer.Regress(self, request, context,   )
BetaPredictionServiceStub.Classify(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
BetaPredictionServiceStub.GetModelMetadata(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
BetaPredictionServiceStub.MultiInference(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
BetaPredictionServiceStub.Predict(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
BetaPredictionServiceStub.Regress(self, request, timeout,  metadata=None, with_call=False, protocol_options=None,  )
PredictionServiceServicer.Classify(self, request, context,   )
PredictionServiceServicer.GetModelMetadata(self, request, context,   )
PredictionServiceServicer.MultiInference(self, request, context,   )
PredictionServiceServicer.Predict(self, request, context,   )
PredictionServiceServicer.Regress(self, request, context,   )
PredictionServiceStub.__init__(self, channel,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\predict_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\regression_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\5.tf-serving\tensorflow_serving\apis\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\6.inception-flasksocketio\app.py
----------------methods----------------

---------------functions---------------
test_connect(  )
test_disconnect(  )
test_live(message,   )
test_message(message,   )


mlmodels\model_tf\raw\tf_serving\6.inception-flasksocketio\camera.py
----------------methods----------------

---------------functions---------------
on_camera_response(  *args)
receive_events_thread(  )


mlmodels\model_tf\raw\tf_serving\6.inception-flasksocketio\inception_utils.py
----------------methods----------------

---------------functions---------------
inception_arg_scope( weight_decay=0.00004, use_batch_norm=True, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, activation_fn=tf.nn.relu, batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,  )


mlmodels\model_tf\raw\tf_serving\6.inception-flasksocketio\inception_v3.py
----------------methods----------------

---------------functions---------------
_reduced_kernel_size_for_small_input(input_tensor, kernel_size,   )
inception_v3(inputs,  num_classes=1000, is_training=True, dropout_keep_prob=0.8, min_depth=16, depth_multiplier=1.0, prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None, create_aux_logits=True, scope="InceptionV3", global_pool=False,  )
inception_v3_base(inputs,  final_endpoint="Mixed_7c", min_depth=16, depth_multiplier=1.0, scope=None,  )


mlmodels\model_tf\raw\tf_serving\6.inception-flasksocketio\object_inception.py
----------------methods----------------

---------------functions---------------
detect_object(img,   )


mlmodels\model_tf\raw\tf_serving\7.object-detection-flask-opencv\opencv.py
----------------methods----------------
Camera.__del__(self,   )
Camera.__init__(self,   )
Camera.get_frame(self,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\7.object-detection-flask-opencv\server.py
----------------methods----------------

---------------functions---------------
gen(camera,   )
index(  )
video_feed(  )


mlmodels\model_tf\raw\tf_serving\7.object-detection-flask-opencv\utils\label_map_util.py
----------------methods----------------

---------------functions---------------
_validate_label_map(label_map,   )
convert_label_map_to_categories(label_map, max_num_classes,  use_display_name=True,  )
create_category_index(categories,   )
create_category_index_from_labelmap(label_map_path,   )
create_class_agnostic_category_index(  )
get_label_map_dict(label_map_path,  use_display_name=False,  )
get_max_label_map_index(label_map,   )
load_labelmap(path,   )


mlmodels\model_tf\raw\tf_serving\7.object-detection-flask-opencv\utils\standard_fields.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\7.object-detection-flask-opencv\utils\string_int_label_map_pb2.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\7.object-detection-flask-opencv\utils\visualization_utils.py
----------------methods----------------

---------------functions---------------
_visualize_boxes(image, boxes, classes, scores, category_index,   **kwargs)
_visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints, category_index,   **kwargs)
_visualize_boxes_and_masks(image, boxes, classes, scores, masks, category_index,   **kwargs)
_visualize_boxes_and_masks_and_keypoints(image, boxes, classes, scores, masks, keypoints, category_index,   **kwargs)
add_cdf_image_summary(values, name,   )
add_hist_image_summary(values, bins, name,   )
draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_boxes_on_image(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_array(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_tensors(images, boxes, classes, scores, category_index,  instance_masks=None, keypoints=None, max_boxes_to_draw=20, min_score_thresh=0.2,  )
draw_keypoints_on_image(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_keypoints_on_image_array(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_mask_on_image_array(image, mask,  color="red", alpha=0.4,  )
draw_side_by_side_evaluation_image(eval_dict, category_index,  max_boxes_to_draw=20, min_score_thresh=0.2,  )
encode_image_array_as_png_str(image,   )
save_image_array_as_png(image, output_path,   )
visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,  instance_masks=None, instance_boundaries=None, keypoints=None, use_normalized_coordinates=False, max_boxes_to_draw=20, min_score_thresh=0.5, agnostic_mode=False, line_thickness=4, groundtruth_box_visualization_color="black", skip_scores=False, skip_labels=False,  )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\app.py
----------------methods----------------

---------------functions---------------
test_connect(  )
test_disconnect(  )
test_live(message,   )
test_message(message,   )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\camera.py
----------------methods----------------

---------------functions---------------
on_camera_response(  *args)
receive_events_thread(  )
run_cam(  )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\face_detection.py
----------------methods----------------

---------------functions---------------
detect_face(img,   )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\standard_fields.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\visual.py
----------------methods----------------

---------------functions---------------
on_camera_response(  *args)
receive_events_thread(  )
run_cam(  )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\visualization_utils.py
----------------methods----------------

---------------functions---------------
_visualize_boxes(image, boxes, classes, scores, category_index,   **kwargs)
_visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints, category_index,   **kwargs)
_visualize_boxes_and_masks(image, boxes, classes, scores, masks, category_index,   **kwargs)
_visualize_boxes_and_masks_and_keypoints(image, boxes, classes, scores, masks, keypoints, category_index,   **kwargs)
add_cdf_image_summary(values, name,   )
add_hist_image_summary(values, bins, name,   )
draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_boxes_on_image(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_array(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_tensors(images, boxes, classes, scores, category_index,  instance_masks=None, keypoints=None, max_boxes_to_draw=20, min_score_thresh=0.2,  )
draw_keypoints_on_image(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_keypoints_on_image_array(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_mask_on_image_array(image, mask,  color="red", alpha=0.4,  )
draw_side_by_side_evaluation_image(eval_dict, category_index,  max_boxes_to_draw=20, min_score_thresh=0.2,  )
encode_image_array_as_png_str(image,   )
save_image_array_as_png(image, output_path,   )
visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,  instance_masks=None, instance_boundaries=None, keypoints=None, use_normalized_coordinates=False, max_boxes_to_draw=20, min_score_thresh=0.5, agnostic_mode=False, line_thickness=4, groundtruth_box_visualization_color="black", skip_scores=False, skip_labels=False,  )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\core\preprocess.py
----------------methods----------------

---------------functions---------------
preprocess_for_eval(image,  out_shape=EVAL_SIZE, data_format="NHWC",  )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\core\visualization.py
----------------methods----------------

---------------functions---------------
bboxes_draw_on_img(img, classes, scores, bboxes,  thickness=2,  )
colors_subselect(colors,  num_classes=21,  )
draw_bbox(img, bbox, shape, label,  color=[255, 0, 0], thickness=2,  )
draw_lines(img, lines,  color=[255, 0, 0], thickness=2,  )
draw_rectangle(img, p1, p2,  color=[255, 0, 0], thickness=2,  )
plt_bboxes(img, classes, scores, bboxes, sess, model, size_image,  distancelabel="m", figsize=(20, 20), linewidth=1.5,  )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\Detection\detector.py
----------------methods----------------
Detector.__init__(self, net_factory, data_size, batch_size, model_path,   )
Detector.predict(self, databatch,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\Detection\fcn_detector.py
----------------methods----------------
FcnDetector.__init__(self, net_factory, model_path,   )
FcnDetector.predict(self, databatch,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\Detection\MtcnnDetector.py
----------------methods----------------
MtcnnDetector.__init__(self, detectors,  min_face_size=25, stride=2, threshold=[0.6, 0.7, 0.7], scale_factor=0.79,  )
MtcnnDetector.calibrate_box(self, bbox, reg,   )
MtcnnDetector.convert_to_square(self, bbox,   )
MtcnnDetector.detect(self, img,   )
MtcnnDetector.detect_face(self, test_data,   )
MtcnnDetector.detect_onet(self, im, dets,   )
MtcnnDetector.detect_pnet(self, im,   )
MtcnnDetector.detect_rnet(self, im, dets,   )
MtcnnDetector.generate_bbox(self, cls_map, reg, scale, threshold,   )
MtcnnDetector.pad(self, bboxes, w, h,   )
MtcnnDetector.processed_image(self, img, scale,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\Detection\nms.py
----------------methods----------------

---------------functions---------------
py_nms(dets, thresh,  mode="Union",  )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\Detection\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\train_models\MTCNN_config.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\train_models\mtcnn_model.py
----------------methods----------------

---------------functions---------------
O_Net(inputs,  label=None, bbox_target=None, landmark_target=None, training=True,  )
P_Net(inputs,  label=None, bbox_target=None, landmark_target=None, training=True,  )
R_Net(inputs,  label=None, bbox_target=None, landmark_target=None, training=True,  )
bbox_ohem(bbox_pred, bbox_target, label,   )
bbox_ohem_orginal(bbox_pred, bbox_target, label,   )
bbox_ohem_smooth_L1_loss(bbox_pred, bbox_target, label,   )
cal_accuracy(cls_prob, label,   )
cls_ohem(cls_prob, label,   )
dense_to_one_hot(labels_dense, num_classes,   )
landmark_ohem(landmark_pred, landmark_target, label,   )
prelu(inputs,   )


mlmodels\model_tf\raw\tf_serving\8.MTCNN-face-detection-flasksocketio-opencv2\train_models\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\app.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\standard_fields.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\visualization_utils.py
----------------methods----------------

---------------functions---------------
_visualize_boxes(image, boxes, classes, scores, category_index,   **kwargs)
_visualize_boxes_and_keypoints(image, boxes, classes, scores, keypoints, category_index,   **kwargs)
_visualize_boxes_and_masks(image, boxes, classes, scores, masks, category_index,   **kwargs)
_visualize_boxes_and_masks_and_keypoints(image, boxes, classes, scores, masks, keypoints, category_index,   **kwargs)
add_cdf_image_summary(values, name,   )
add_hist_image_summary(values, bins, name,   )
draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_box_on_image_array(image, ymin, xmin, ymax, xmax,  color="red", thickness=4, display_str_list=(), use_normalized_coordinates=True,  )
draw_bounding_boxes_on_image(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_array(image, boxes,  color="red", thickness=4, display_str_list_list=(),  )
draw_bounding_boxes_on_image_tensors(images, boxes, classes, scores, category_index,  instance_masks=None, keypoints=None, max_boxes_to_draw=20, min_score_thresh=0.2,  )
draw_keypoints_on_image(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_keypoints_on_image_array(image, keypoints,  color="red", radius=2, use_normalized_coordinates=True,  )
draw_mask_on_image_array(image, mask,  color="red", alpha=0.4,  )
draw_side_by_side_evaluation_image(eval_dict, category_index,  max_boxes_to_draw=20, min_score_thresh=0.2,  )
encode_image_array_as_png_str(image,   )
save_image_array_as_png(image, output_path,   )
visualize_boxes_and_labels_on_image_array(image, boxes, classes, scores, category_index,  instance_masks=None, instance_boundaries=None, keypoints=None, use_normalized_coordinates=False, max_boxes_to_draw=20, min_score_thresh=0.5, agnostic_mode=False, line_thickness=4, groundtruth_box_visualization_color="black", skip_scores=False, skip_labels=False,  )


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\core\preprocess.py
----------------methods----------------

---------------functions---------------
preprocess_for_eval(image,  out_shape=EVAL_SIZE, data_format="NHWC",  )


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\core\visualization.py
----------------methods----------------

---------------functions---------------
bboxes_draw_on_img(img, classes, scores, bboxes,  thickness=2,  )
colors_subselect(colors,  num_classes=21,  )
draw_bbox(img, bbox, shape, label,  color=[255, 0, 0], thickness=2,  )
draw_lines(img, lines,  color=[255, 0, 0], thickness=2,  )
draw_rectangle(img, p1, p2,  color=[255, 0, 0], thickness=2,  )
plt_bboxes(img, classes, scores, bboxes, sess, model, size_image,  distancelabel="m", figsize=(20, 20), linewidth=1.5,  )


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\Detection\detector.py
----------------methods----------------
Detector.__init__(self, net_factory, data_size, batch_size, model_path,   )
Detector.predict(self, databatch,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\Detection\fcn_detector.py
----------------methods----------------
FcnDetector.__init__(self, net_factory, model_path,   )
FcnDetector.predict(self, databatch,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\Detection\MtcnnDetector.py
----------------methods----------------
MtcnnDetector.__init__(self, detectors,  min_face_size=25, stride=2, threshold=[0.6, 0.7, 0.7], scale_factor=0.79,  )
MtcnnDetector.calibrate_box(self, bbox, reg,   )
MtcnnDetector.convert_to_square(self, bbox,   )
MtcnnDetector.detect(self, img,   )
MtcnnDetector.detect_face(self, test_data,   )
MtcnnDetector.detect_onet(self, im, dets,   )
MtcnnDetector.detect_pnet(self, im,   )
MtcnnDetector.detect_rnet(self, im, dets,   )
MtcnnDetector.generate_bbox(self, cls_map, reg, scale, threshold,   )
MtcnnDetector.pad(self, bboxes, w, h,   )
MtcnnDetector.processed_image(self, img, scale,   )

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\Detection\nms.py
----------------methods----------------

---------------functions---------------
py_nms(dets, thresh,  mode="Union",  )


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\Detection\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\train_models\MTCNN_config.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\train_models\mtcnn_model.py
----------------methods----------------

---------------functions---------------
O_Net(inputs,  label=None, bbox_target=None, landmark_target=None, training=True,  )
P_Net(inputs,  label=None, bbox_target=None, landmark_target=None, training=True,  )
R_Net(inputs,  label=None, bbox_target=None, landmark_target=None, training=True,  )
bbox_ohem(bbox_pred, bbox_target, label,   )
bbox_ohem_orginal(bbox_pred, bbox_target, label,   )
bbox_ohem_smooth_L1_loss(bbox_pred, bbox_target, label,   )
cal_accuracy(cls_prob, label,   )
cls_ohem(cls_prob, label,   )
dense_to_one_hot(labels_dense, num_classes,   )
landmark_ohem(landmark_pred, landmark_target, label,   )
prelu(inputs,   )


mlmodels\model_tf\raw\tf_serving\9.MTCNN-face-detection-opencv\train_models\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\model_tf\rl\0_template_rl.py
----------------methods----------------
Agent.__init__(self, history, do_action,  params={},  )
Agent.discount_rewards(self, r,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,  state=None, history=None, reward=None,  )
Agent.predict_action(self, inputs,   )
Agent.run_sequence(self, history, do_action, params,   )
Agent.train(self,  n_iters=1, n_log_freq=1, state_initial=None, reward_initial=None,  )
Model.__init__(self, history,  params={},  )

---------------functions---------------
do_action_example(action_dict,   )
fit(model, df, do_action,  state_initial=None, reward_initial=None, params=None,  )
predict(model, sess, df,  do_action=None, params=params,  )
val(x, y,   )


mlmodels\model_tf\rl\1.turtle-agent.py
----------------methods----------------

---------------functions---------------
buy_stock(real_movement, signal,  initial_money=10000, max_buy=20, max_sell=20,  )


mlmodels\model_tf\rl\10.duel-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip, batch_size,   )
Agent.act(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.replay(self, batch_size,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\rl\11.double-duel-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, input_size, output_size, layer_size, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\rl\12.duel-recurrent-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\rl\13.double-duel-recurrent-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, input_size, output_size, layer_size, learning_rate, name,   )

---------------functions---------------


mlmodels\model_tf\rl\14.actor-critic-agent.py
----------------methods----------------
Actor.__init__(self, name, input_size, output_size, size_layer,   )
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories_and_train(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\rl\15.actor-critic-duel-agent.py
----------------methods----------------
Actor.__init__(self, name, input_size, output_size, size_layer,   )
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories_and_train(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\rl\16.actor-critic-recurrent-agent.py
----------------methods----------------
Actor.__init__(self, name, input_size, output_size, size_layer,   )
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories_and_train(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\rl\17.actor-critic-duel-recurrent-agent.py
----------------methods----------------
Actor.__init__(self, name, input_size, output_size, size_layer,   )
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories_and_train(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Critic.__init__(self, name, input_size, output_size, size_layer, learning_rate,   )

---------------functions---------------


mlmodels\model_tf\rl\18.curiosity-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\rl\19.recurrent-curiosity-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done, rnn_state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\rl\2.moving-average-agent.py
----------------methods----------------

---------------functions---------------
buy_stock(real_movement, signal,  initial_money=10000, max_buy=20, max_sell=20,  )


mlmodels\model_tf\rl\20.duel-curiosity-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\rl\21.neuro-evolution-agent.py
----------------methods----------------
NeuroEvolution.__init__(self, population_size, mutation_rate, model_generator, state_size, window_size, trend, skip, initial_money,   )
NeuroEvolution._initialize_population(self,   )
NeuroEvolution.act(self, p, state,   )
NeuroEvolution.buy(self, individual,   )
NeuroEvolution.calculate_fitness(self,   )
NeuroEvolution.crossover(self, parent1, parent2,   )
NeuroEvolution.evolve(self,  generations=20, checkpoint=5,  )
NeuroEvolution.get_state(self, t,   )
NeuroEvolution.inherit_weights(self, parent, child,   )
NeuroEvolution.mutate(self, individual,  scale=1.0,  )
neuralnetwork.__init__(self, id_,  hidden_size=128,  )

---------------functions---------------
feed_forward(X, nets,   )
relu(X,   )
softmax(X,   )


mlmodels\model_tf\rl\22.neuro-evolution-novelty-search-agent.py
----------------methods----------------
NeuroEvolution.__init__(self, population_size, mutation_rate, model_generator, state_size, window_size, trend, skip, initial_money,   )
NeuroEvolution._initialize_population(self,   )
NeuroEvolution._memorize(self, q, i, limit,   )
NeuroEvolution.act(self, p, state,   )
NeuroEvolution.buy(self, individual,   )
NeuroEvolution.calculate_fitness(self,   )
NeuroEvolution.crossover(self, parent1, parent2,   )
NeuroEvolution.evaluate(self, individual, backlog, pop,  k=4,  )
NeuroEvolution.evolve(self,  generations=20, checkpoint=5,  )
NeuroEvolution.get_state(self, t,   )
NeuroEvolution.inherit_weights(self, parent, child,   )
NeuroEvolution.mutate(self, individual,  scale=1.0,  )
neuralnetwork.__init__(self, id_,  hidden_size=128,  )

---------------functions---------------
feed_forward(X, nets,   )
relu(X,   )
softmax(X,   )


mlmodels\model_tf\rl\3.signal-rolling-agent.py
----------------methods----------------

---------------functions---------------
buy_stock(real_movement,  delay=5, initial_state=1, initial_money=10000, max_buy=20, max_sell=20,  )


mlmodels\model_tf\rl\4.policy-gradient-agent_old.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent.buy(self, initial_money,   )
Agent.discount_rewards(self, r,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,  reward_state=None,  )
Agent.predict(self, inputs,   )
Agent.predict_sequence(self, trend_input, do_action,  param=None,  )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward,   )

---------------functions---------------
do_action_example(action_dict,   )
fit(model, df, do_action,   )
predict(model, sess, df, do_action,   )
test( filename='dataset/GOOG-year.csv',  )


mlmodels\model_tf\rl\4_policy-gradient-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent.discount_rewards(self, r,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,  reward_state=None,  )
Agent.predict(self, inputs,   )
Agent.predict_sequence(self, pars,  trend_history=None,  )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward,  checkpoint=10,  )

---------------functions---------------
fit(model, dftrain,  params={},  )
predict(model, sess, dftest,  params={},  )
test( filename='dataset/GOOG-year.csv',  )


mlmodels\model_tf\rl\5_q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip, batch_size,   )
Agent.act(self, state,   )
Agent.get_state(self, t,   )
Agent.predict_sequence(self, pars,  trend_history=None,  )
Agent.replay(self, batch_size,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, state_size, window_size, trend, skip, iterations, initial_reward,  checkpoint=10,  )

---------------functions---------------
fit(model, dftrain,  params={},  )
predict(model, sess, dftest,  params={},  )
test( filename='../dataset/GOOG-year.csv',  )


mlmodels\model_tf\rl\6_evolution-strategy-agent.py
----------------methods----------------
Agent.__init__(self, model, window_size, trend, skip, initial_money,   )
Agent.act(self, sequence,   )
Agent.fit(self, iterations, checkpoint,   )
Agent.get_reward(self, weights,   )
Agent.get_state(self, t,   )
Agent.run_sequence(self, df_test,   )
Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate,   )
Deep_Evolution_Strategy._get_weight_from_population(self, weights, population,   )
Deep_Evolution_Strategy.get_weights(self,   )
Deep_Evolution_Strategy.train(self,  epoch=100, print_every=1,  )
Model.__init__(self, input_size, layer_size, output_size, window_size, skip, initial_money,  iterations=500, checkpoint=10,  )
Model.get_weights(self,   )
Model.predict(self, inputs,   )
Model.set_weights(self, weights,   )

---------------functions---------------
fit(model, dftrain,  params={},  )
get_imports(  )
predict(model, sess, dftest,  params={},  )
test( filename='../dataset/GOOG-year.csv',  )


mlmodels\model_tf\rl\7.double-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, done,   )
Agent._select_action(self, state,   )
Agent.get_predicted_action(self, sequence,   )
Agent.get_state(self, t,   )
Agent.predict(self, inputs,   )
Agent.run_sequence(self, initial_money,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, window_size, trend, skip, iterations, initial_reward,  checkpoint=10,  )
QModel.__init__(self, input_size, output_size, layer_size, learning_rate,   )

---------------functions---------------
fit(model, dftrain,  params={},  )
predict(model, sess, dftest,  params={},  )
test( filename='../dataset/GOOG-year.csv',  )


mlmodels\model_tf\rl\8.recurrent-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )

---------------functions---------------


mlmodels\model_tf\rl\9.double-recurrent-q-learning-agent.py
----------------methods----------------
Agent.__init__(self, state_size, window_size, trend, skip,   )
Agent._assign(self, from_name, to_name,   )
Agent._construct_memories(self, replay,   )
Agent._memorize(self, state, action, reward, new_state, dead, rnn_state,   )
Agent._select_action(self, state,   )
Agent.buy(self, initial_money,   )
Agent.get_state(self, t,   )
Agent.train(self, iterations, checkpoint, initial_money,   )
Model.__init__(self, input_size, output_size, layer_size, learning_rate, name,   )

---------------functions---------------


mlmodels\model_tf\rl\updated-NES-google.py
----------------methods----------------
Agent.__init__(self, model, money, max_buy, max_sell, close, window_size, skip,   )
Agent.act(self, sequence,   )
Agent.buy(self,   )
Agent.fit(self, iterations, checkpoint,   )
Agent.get_reward(self, weights,   )
Deep_Evolution_Strategy.__init__(self, weights, reward_function, population_size, sigma, learning_rate,   )
Deep_Evolution_Strategy._get_weight_from_population(self, weights, population,   )
Deep_Evolution_Strategy.get_weights(self,   )
Deep_Evolution_Strategy.train(self,  epoch=100, print_every=1,  )
Model.__init__(self, input_size, layer_size, output_size,   )
Model.get_weights(self,   )
Model.predict(self, inputs,   )
Model.set_weights(self, weights,   )

---------------functions---------------
act(model, sequence,   )
f(w,   )
get_state(data, t, n,   )


mlmodels\model_tf\rl\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\preprocess\generic.py
----------------methods----------------
pandasDataset.__init__(self,  root="", train=True, transform=None, target_transform=None, download=False, data_pars=None,  )

---------------functions---------------
get_dataset_torch(data_pars,   )
load_function( uri_name="path_norm",  )
tf_dataset(dataset_pars,   )
torch_datasets_wrapper(sets,  args_list=None,  **args)


mlmodels\preprocess\image.py
----------------methods----------------

---------------functions---------------
torch_transform_mnist(  )
torchvision_dataset_MNIST_load(path,   **args)


mlmodels\preprocess\tabular.py
----------------methods----------------

---------------functions---------------


mlmodels\preprocess\tabular_keras.py
----------------methods----------------

---------------functions---------------
check_model(model, model_name, x, y,  check_model_io=True,  )
gen_sequence(dim, max_len, sample_size,   )
get_test_data( sample_size=1000, embedding_size=4, sparse_feature_num=1, dense_feature_num=1, sequence_feature=None, classification=True, include_length=False, hash_flag=False, prefix='', use_group=False,  )
get_xy_fd_dien( use_neg=False, hash_flag=False,  )
get_xy_fd_din( hash_flag=False,  )
get_xy_fd_dsin( hash_flag=False,  )
has_arg(fn, name,  accept_all=False,  )
layer_test(layer_cls,  kwargs={}, input_shape=None, input_dtype=None, input_data=None, expected_output=None, expected_output_dtype=None, fixed_batch_size=False, supports_masking=False,  )


mlmodels\preprocess\text.py
----------------methods----------------

---------------functions---------------


mlmodels\preprocess\text_keras.py
----------------methods----------------
Preprocess_namentity.__init__(self, max_len,   **args)
Preprocess_namentity.compute(self, df,   )
Preprocess_namentity.get_data(self,   )

---------------functions---------------


mlmodels\preprocess\text_torch.py
----------------methods----------------

---------------functions---------------
clean_str(string,   )
imdb_spacy_tokenizer(text,  lang="en",  )
test_onehot_sentences(data, max_len,   )
test_pandas_fillna(data,   **args)
test_word_categorical_labels_per_sentence(data, max_len,   )
test_word_count(data,   )


mlmodels\preprocess\ztemp.py
----------------methods----------------
DataFrameBucketIterator.data(self,   )
DataFrameDataset.__init__(self, data_pars,   )
DataFrameDataset.__iter__(self,   )
DataFrameDataset.__len__(self,   )
DataFrameDataset.shuffle(self,  random_state=None,  )
DataFrameIterator.__init__(self, df,   )
MNIST.__getitem__(self, index,   )
MNIST.__init__(self, root,  train=True, transform=None, target_transform=None, download=False,  )
MNIST.__len__(self,   )
MNIST.class_to_idx(self,   )
MNIST.processed_folder(self,   )
MNIST.raw_folder(self,   )
MNIST.test_data(self,   )
MNIST.test_labels(self,   )
MNIST.train_data(self,   )
MNIST.train_labels(self,   )
NewsDataLoader.__init__(self, path,  max_src_len=100, field=None, debug=False,  **kwargs)
NewsDataset.data(self,   )

---------------functions---------------
batch_generator(iterable,  n=1,  )
custom_dataset(  )
get_loader(fix_length, vocab_threshold, batch_size,   )
image_dir_load(path,   )
pandas_dataset(  )
pickle_load(file,   )
text_dataloader(  )


mlmodels\preprocess\__init__.py
----------------methods----------------

---------------functions---------------


mlmodels\template\00_template_keras.py
----------------methods----------------
Model.__init__(self,  model_pars=None, data_pars=None, compute_pars=None,  **kwargs)
Model_empty.__init__(self,  model_pars=None, compute_pars=None,  )

---------------functions---------------
_preprocess_XXXX(df,   **kw)
fit(model,  session=None, data_pars=None, model_pars=None, compute_pars=None, out_pars=None,  **kwargs)
get_dataset(  **kw)
get_params( choice=0, data_path="dataset/",  **kw)
load(path,   )
log( n=0, m=1,  *s)
metrics(ypred, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
os_package_root_path(filepath,  sublevel=0, path_add="",  )
path_setup( out_folder="", sublevel=0, data_path="dataset/",  )
predict(model, data_pars,  compute_pars=None, out_pars=None,  **kwargs)
reset_model(  )
save(model, path,   )
test( data_path="dataset/", pars_choice=0,  )


mlmodels\template\model_xxx.py
----------------methods----------------

---------------functions---------------
fit(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
fit_metrics(model,  data_pars=None, compute_pars=None, out_pars=None,  **kw)
get_dataset( data_pars=None,  **kw)
get_params( param_pars={},  **kw)
load( load_pars={},  )
predict(model,  sess=None, data_pars=None, compute_pars=None, out_pars=None,  **kw)
reset_model(  )
save( model=None, session=None, save_pars={},  )
test( data_path="dataset/", pars_choice="json", config_mode="test",  )


mlmodels\template\zarchive\gluonts_model.py
----------------methods----------------
Model.__init__(self,  model_pars=None, compute_pars=None,  )

---------------functions---------------
get_params( choice=0, data_path="dataset/",  **kw)
test( data_path="dataset/",  )
test2( data_path="dataset/", out_path="GLUON/gluon.png", reset=True,  )


mlmodels\template\zarchive\model_tf_sequential.py
----------------methods----------------

---------------functions---------------
fit(model, data_pars,  out_pars=None, compute_pars=None,  **kwargs)
get_dataset( data_pars=None,  )
get_pars( choice="test",  **kwargs)
log( n=0, m=1,  *s)
metrics(model,  sess=None, data_pars=None, out_pars=None,  )
os_file_path(data_path,   )
os_module_path(  )
os_package_root_path(filepath,  sublevel=0, path_add="",  )
predict(model, sess,  data_pars=None, out_pars=None, compute_pars=None, get_hidden_state=False, init_value=None,  )
reset_model(  )
test( data_path="dataset/GOOG-year.csv", out_path="", reset=True,  )
test2( data_path="dataset/GOOG-year.csv",  )


mlmodels\ztest\benchmark3.py
----------------methods----------------

---------------functions---------------
benchmark_run( bench_pars=None, args=None, config_mode="test",  )
benchmark_run_mnist( bench_pars=None, args=None, config_mode="test",  )
cli_load_arguments( config_file=None,  )
config_model_list( folder=None,  )
get_all_json_path(json_path,   )
main(  )
metric_eval( actual=None, pred=None, metric_name="mean_absolute_error",  )
preprocess_timeseries_m5( data_path=None, dataset_name=None, pred_length=10, item_id=None,  )

