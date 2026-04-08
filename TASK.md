Objective: a prototype agentic ai system
一个原型版：不需要很完美，能跑 以及展示核心功能
智能体式AI系统：
    传统AI：输入 -> 模型 -> 输出
    Agentic AI: 会想下一步要干什么，决策（policy） 下一步要干什么。 loop 循环 观察 决策 行动。 tools 工具 调用模型 函数 API。 记忆 memory 记住过去的信息。 反思reflection 根据结构调整策略

ability: 
    analyze multivariate health time series data  -- 多个指标 随着时间变化记录 心率 血压 血氧 呼吸
    make adaptive decisions  -- 根据情况的变化，动态调整决策，不能是只要心率大于多少 就报警 比如 数据稳定 就不用模型 数据突然异常 调用预测模型 如果不确定 再用别的工具 连续几次错误 改变策略
    optionally invoke tools: models, APIs, or functions to improve predictions or insights. 不一定每一次都需要调用工具 需要的时候再用 model 就是我自己训练的模型 函数 就是普通代码函数 计算平均值， 检测异常，计算不确定性，接口 调用外部服务， 比如 LLM 医疗知识库， 远程模拟服务

steps:
    观察数据: 心率有点奇怪
    决策：我要不要用工具
    调用：调用function 工具1 anomaly score = detect_anomaly(data)
    如果异常高：调用工具2 model 预测 risk = LSTM 预测的未来 
    如果还是不确定: 调用API 工具3 让LLM解释情况
    做决策: call 999 or not。

Tasks：early warning system for 高血糖 低血糖

dataset: OhioT1DM 数据 CGM insulin HR exercise meal

a agent loop: include policy, and a tool interface (at least two tool)
    policy: 决策 rule-based
    a tool interface: tool 1 and tool 2

tool 2: a forecasting model: GRU

tool 1: anomaly detector 

non-trivial tool use: 有策略 有理由的工具使用 而不是机械调用（不是每一步都调用模型，在特定的情况下调用工具)
    call a forecasting model only when uncertainty is high
    遇到没见过的数据（out-of-distribution signals)-OOD 这种血糖模式从来没见过 去查资料 或者 历史数据
    trigger explanation or summaries when appropriate -- print explanation or LLM 进阶

evaluation: one for predictive performance, one for agent behaviour quality, including when and how tools are used.


步骤: 
1. 观察数据 （CGM + 其他变量)
2. 判断情况 稳定 / 异常 / 不确定
3. 做决策 要不要用模型，要不要看更多变量 --cgm稳定 不看其他变量 cgm 低 看看 insulin 判断是不是打了胰岛素
4. 调用工具 如果需要:
    lstm 预测 未来血糖
    anamaly detector 异常检测 -- 判断现在是不是不正常，比如血糖突然下降，波动很大，简单规则 or 模型 
    retrieval system 检索系统 -- 去查资料 查历史数据，找类似病人 或者调用LLM 比如OpenAI API 
5. 行动 发出预警 or 不做事
6. 更新记忆 之前做的决策 准不准



