from src.helper import convert_enc_to_string


def get_results_dict(tokenizer, strategy, inputs, outputs, idxs, i, batch, outer_it, both_dir=False):
    alloutputs = []

    for j, ele in enumerate(inputs['input_ids']):
        # pdb.set_trace()
        entropy, margin, gradient_embed = None, None, None
        string = convert_enc_to_string(ele, tokenizer)
        prediction = outputs[i]['test_pred'][j].item()
        confidence = outputs[i]['test_conf'][j].item()
        if strategy == 'entropy':
            entropy = outputs[i]['entropy'][j].item()
        if strategy == 'margin':
            margin = outputs[i]['margin'][j].item()
        if strategy == 'badge':
            gradient_embed = outputs[i]['gradient_embed'][j]
        dict_res = {
            'idx': idxs[j].item(),
            'sentence': string,
            'prediction': prediction,
            'confidence': confidence,
            'entropy': entropy,
            'margin': margin,
            'gradient_embed': gradient_embed
        }
        if outer_it != -1:
            dict_res['outer_it'] = outer_it
        if 'labels' in batch:
            ground_truth = batch['labels'][j]
            dict_res['label'] = ground_truth.item()

        alloutputs.append(dict_res)

    if both_dir:
        newoutputs = []
        assert len(alloutputs) % 2 == 0
        halfsize = int(len(alloutputs)//2)
        for i in range(halfsize):
            dict_res_new = {
                'idx1': alloutputs[i]['idx'],
                'idx2': alloutputs[i+halfsize]['idx'],
                'sentence1': alloutputs[i]['sentence'],
                'sentence2': alloutputs[i+halfsize]['sentence'],
                'prediction1': alloutputs[i]['prediction'],
                'prediction2': alloutputs[i+halfsize]['prediction'],
                'confidence1': alloutputs[i]['confidence'],
                'confidence2': alloutputs[i+halfsize]['confidence'],
                'entropy1': alloutputs[i]['entropy'],
                'entropy2': alloutputs[i+halfsize]['entropy'],
                'margin1': alloutputs[i]['margin'],
                'margin2': alloutputs[i+halfsize]['margin'],
                'gradient_embed1': alloutputs[i]['gradient_embed'],
                'gradient_embed2': alloutputs[i+halfsize]['gradient_embed']
            }
            if 'label' in alloutputs[i]:
                dict_res_new['label'] = alloutputs[i]['label']
            if 'outer_it' in alloutputs[i]:
                dict_res_new['outer_it'] = alloutputs[i]['outer_it']

            newoutputs.append(dict_res_new)
        alloutputs = newoutputs

    return alloutputs
