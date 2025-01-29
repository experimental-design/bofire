import pandas as pd


# this are the default generators used for fractional factorial designs in BoFire
default_fracfac_generators = pd.DataFrame.from_dict(
    [
        {"n_factors": 3, "n_generators": 1, "generator": "C = AB"},
        {"n_factors": 4, "n_generators": 1, "generator": "D = ABC"},
        {"n_factors": 5, "n_generators": 1, "generator": "E = ABCD"},
        {"n_factors": 5, "n_generators": 2, "generator": "D= AB ; E=  AC"},
        {"n_factors": 6, "n_generators": 1, "generator": "F = ABCDE"},
        {"n_factors": 6, "n_generators": 2, "generator": "E = ABC ; F = BCD"},
        {"n_factors": 6, "n_generators": 3, "generator": "D= AB ; E=  AC ; F= BC"},
        {"n_factors": 7, "n_generators": 1, "generator": "G = ABCDEF"},
        {"n_factors": 7, "n_generators": 2, "generator": "F = ABCD ; G = ABDE"},
        {"n_factors": 7, "n_generators": 3, "generator": "F = ABC ; F = BCD ; G = ACD"},
        {
            "n_factors": 7,
            "n_generators": 4,
            "generator": "D = AB ; E= AC ; F = BC ; G = ABC",
        },
        {"n_factors": 8, "n_generators": 1, "generator": "H = ABCDEFG"},
        {"n_factors": 8, "n_generators": 2, "generator": "G = ABCD ; H = ABEF"},
        {
            "n_factors": 8,
            "n_generators": 3,
            "generator": "F = ABC ; G = ABD ; H = BCDE",
        },
        {
            "n_factors": 8,
            "n_generators": 4,
            "generator": "E = BCD ; F = ACD ; G = ABC ; H = ABD ",
        },
        {"n_factors": 9, "n_generators": 2, "generator": "H = ACDFG ; J = BCEFG"},
        {
            "n_factors": 9,
            "n_generators": 3,
            "generator": "G = ABCD ; H = ACEF ; J = CDEF",
        },
        {
            "n_factors": 9,
            "n_generators": 4,
            "generator": "F = BCDE ; G = ACDE ; H = ABDE ; J = ABCE",
        },
        {
            "n_factors": 9,
            "n_generators": 5,
            "generator": "E = ABC ; F = BCD ; G = ACD ; H = ABD ; J = ABCD ",
        },
        {
            "n_factors": 10,
            "n_generators": 3,
            "generator": "H = ABCG ; J = BCDE ; K = ACDF",
        },
        {
            "n_factors": 10,
            "n_generators": 4,
            "generator": "G = BCDF ; H = ACDF ; J = ABDE ; K = ABCE ",
        },
        {
            "n_factors": 10,
            "n_generators": 5,
            "generator": "F = ABCD ; G = ABCE ; H = ABDE ; J = ACDE ; K = BCDE ",
        },
        {
            "n_factors": 10,
            "n_generators": 6,
            "generator": "E = ABC ; F = BCD ; G = ACD ; H = ABD ; J = ABCD ; K = AB ",
        },
        {
            "n_factors": 11,
            "n_generators": 4,
            "generator": "H = ABCG ; J = BCDE ; K = ACDF ; L = ABCDEFG",
        },
        {
            "n_factors": 11,
            "n_generators": 5,
            "generator": "G = CDE ; H = ABCD ; J = ABF ; K = BDEF ; L = ADEF ",
        },
        {
            "n_factors": 11,
            "n_generators": 6,
            "generator": "F = ABC ; G = BCD ; H = CDE ; J = ACD ; K = ADE ; L = BDE ",
        },
        {
            "n_factors": 11,
            "n_generators": 7,
            "generator": "E = ABC ; F = BCD ; G = ACD ; H = ABD ; J = ABCD ; K = AB ; L = AC ",
        },
        {
            "n_factors": 12,
            "n_generators": 5,
            "generator": "H = ACDG ; J = ABCD ; K = BCFG ; L = ABDEFG ; M = CDEF",
        },
        {
            "n_factors": 12,
            "n_generators": 6,
            "generator": "G = DEF ; H = ABC ; J = BCDE ; K = BCDF ; L = ABEF ; M = ACEF",
        },
        {
            "n_factors": 12,
            "n_generators": 7,
            "generator": "F = ACE ; G = ACD ; H = ABD ; J = ABE ; K = CDE ; L = ABCDE ; M = ADE ",
        },
        {
            "n_factors": 12,
            "n_generators": 8,
            "generator": "E = ABC ; F = ABD ; G = ACD ; H = BCD ; J = ABCD ; K = AB ; L = AC ; M = AD ",
        },
        {
            "n_factors": 13,
            "n_generators": 6,
            "generator": "H = DEFG ; J = BCEG ; K = BCDFG ; L = ABDEF ; M = ACEF ; N = ABC ",
        },
        {
            "n_factors": 13,
            "n_generators": 7,
            "generator": "G = ABC ; H = DEF ; J = BCDF ; K = BCDE ; L = ABEF ; M = ACEF ; N = BCEF ",
        },
        {
            "n_factors": 13,
            "n_generators": 8,
            "generator": "F = ACE ; G = BCE ; H = ABC ; J = CDE ; K = ABCDE ; L = ABE ; M = ACD ; N = ADE ",
        },
        {
            "n_factors": 13,
            "n_generators": 9,
            "generator": "E = ABC ; F = ABD ; G = ACD ; H = BCD ; J = ABCD ; K = AB ; L = AC ; M = AD ; N = BC ",
        },
        {
            "n_factors": 14,
            "n_generators": 7,
            "generator": "H = EFG ; J = BCFG ; K = BCEG ; L = ABEF ; M = ACEF ; N = BCDEF ; O = ABC ",
        },
        {
            "n_factors": 14,
            "n_generators": 8,
            "generator": "G = BEF ; H = BCF ; J = DEF ; K = CEF ; L = BCE ; M = CDF ; N = ACDE ; O = BCDEF ",
        },
        {
            "n_factors": 14,
            "n_generators": 9,
            "generator": "F = ABC ; G = ABD ; H = ABE ; J = ACD ; K = ACE ; L = ADE ; M = BCD ; N = BCE ; O = BDE ",
        },
        {
            "n_factors": 14,
            "n_generators": 10,
            "generator": "E = ABC ; F = ABD ; G = ACD ; H = BCD ; J = ABCD ; K = AB ; L = AC ; M = AD ; N = BC ; O = BD ",
        },
        {
            "n_factors": 15,
            "n_generators": 8,
            "generator": "H = ABFG ; J = ACDEF ; K = BEF ; L = ABCEG ; M = CDFG ; N = ACDEG ; O = EFG ; P = ABDEFG ",
        },
        {
            "n_factors": 15,
            "n_generators": 9,
            "generator": "G = ABC ; H = ABD ; J = ABE ; K = BCDE ; L = ACF ; M = ADF ; N = AEF ; O = CDEF ; P = ABCDEF",
        },
        {
            "n_factors": 15,
            "n_generators": 10,
            "generator": "F = ABC ; G = ABD ; H = ABE ; J = ACD ; K = ACE ; L = ADE; M = BCD ; N = BCE ; O = BDE; P = CDE",
        },
        {
            "n_factors": 15,
            "n_generators": 11,
            "generator": "E = ABC ; F = ABD ; G = ACD ; H = BCD ; J = ABCD ; K = AB ; L = AC ; M = AD ; N = BC ; O = BD ; P = CD ",
        },
    ],
)
