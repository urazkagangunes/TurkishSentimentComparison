from src.hisnet.hisnet_classifier import (
    HisNetClassifier
)

from src.hisnet.hisnet_dependency_classifier import (
    HisNetDependencyClassifier
)


TEST_SENTENCES = [
    "..saçma bir konuyu nasılda filim yapmışlar maşallah",

    "belki de insanlara bu filmi sevdiren onların bu filmde rol almasıdır",

    "hayatım boyumca bir çok film izledim en heyecan verici film titanik izlerken hala ilk heyacın tadıyorum",

    "Hayatımda beni etkisi altına alan en iyi filmlerden biri",

    "izlediğim en iyi filmlerden biri",

    "keşke ölmeseydi jack en sonunda onu hatırlayınca çok üzülüyorum ve kendimi geminin içinde hissediyorum",

    "bence tıtanıc hayatımda izlediğim en ama en güzel flim defalarca izledim ve defalarca izlerim tıtanıcı ve leonardo dı caprio yu ççççççççççççoooooooooooooooooooooooook seviyorum onun bir numaralı hayranıyım",

    "cok güzel bir film hele 97 filmiyse harika",

    "ben ikisinin ortası diyorum",

    "izlediğim en başarısız filmlerden biri"
]


hisnet = (
    HisNetClassifier()
)

dependency = (
    HisNetDependencyClassifier()
)


for index, text in enumerate(
    TEST_SENTENCES,
    start=1
):

    print(
        "\n"
        + "=" * 90
    )

    print(
        f"TEST {index}"
    )

    print(
        text
    )

    hisnet_result = (
        hisnet.classify(
            text
        )
    )

    dependency_result = (
        dependency.classify(
            text,
            hisnet_result=hisnet_result
        )
    )

    initial_positive = sum(
        scope[
            "original_positive"
        ]
        for scope in dependency_result[
            "scopes"
        ]
    )

    initial_negative = sum(
        scope[
            "original_negative"
        ]
        for scope in dependency_result[
            "scopes"
        ]
    )

    print(
        "HisNet:",
        hisnet_result[
            "positive_score"
        ],
        hisnet_result[
            "negative_score"
        ],
        hisnet_result[
            "prediction"
        ]
    )

    print(
        "Dependency initial:",
        initial_positive,
        initial_negative
    )

    print(
        "Dependency final:",
        dependency_result[
            "positive_score"
        ],
        dependency_result[
            "negative_score"
        ],
        dependency_result[
            "prediction"
        ]
    )

    print(
        "Baseline reused:",
        dependency_result[
            "baseline_reused"
        ]
    )

    print(
        "Operations:",
        [
            operation
            for scope in dependency_result[
                "scopes"
            ]
            for operation in scope.get(
                "operations",
                []
            )
        ]
    )

    assert abs(
        initial_positive
        - hisnet_result[
            "positive_score"
        ]
    ) < 1e-9

    assert abs(
        initial_negative
        - hisnet_result[
            "negative_score"
        ]
    ) < 1e-9


print(
    "\n"
    + "=" * 90
)

print(
    "ALL FAIRNESS TESTS PASSED"
)

print(
    "=" * 90
)