### 这是我们测评的工作之一：

Qwen2.5-7B是我们需要测评的大模型，我们通过要求Qwen2.5-7B根据中文歧义句生成多句英文翻译，再对检验生成的英文翻译是否能够表达出中文的不同歧义意思，从而判断Qwen能否理解该句中文歧义句。

```
例如：
中文歧义句为：洗白衣服。
歧义原因为：短语存在结构切分歧义：'洗白/衣服'(使衣服变白) 与 '洗/白衣服'(清洗白色衣物)两种切分均可成立
英文翻译为： [
            "Whitening clothes",
            "Whitening clothes",
            "Washing white clothes",
            "Whiten clothes",
            "Wash white clothes"
        ]
那么 "Washing white clothes"表达出 '洗/白衣服'，"Whitening clothes",表达出'洗白/衣服'，因此Qwen2.5-7B能够理解该句歧义
```

我们会使用大模型（DeepSeek-R1, GPT-o3-mini）辅助我们完成这一步骤。

调用大模型针对性的写prompt，将中文歧义句（source）、歧义原因（reasons）、英文翻译（output_processed）提供给大模型，要求其生成英文翻译句是否表达出了中文的歧义原因（ambiguity），大模型判断的原因（reasons），并与英文翻译（output_processed）checkAmbiguity_V1+V2+NP.json的结构结合（original)。输出如下图所示

![](.\output.png)

*Qwen2.5-7B-V1+V2+NP.json中含有：V1+V2+NP类型的歧义句（source）、歧义结构类型（amb_type）以及英文翻译（output_processed）
checkAmbiguity_V1+V2+NP.json中含有V1+V2+NP类型歧义句：人工判断的该句是否有歧义（ambiguity）、歧义的原因（reasons）、例子（examples）、中文歧义句（source）*

