 
!(register-module! ../../../hyperon-openpsi) ;to load Relative path for current page
!(register-module! ../../psi-utilities)
!(import! &self hyperon-openpsi:psi-utilities:normalization)

; Test sigmoid function
!(assertEqual (sigmoid 0) 0.5)
!(assertEqual (sigmoid 1) 0.7310585786300049)
!(assertEqual (sigmoid -1) 0.2689414213699951)

; Test normalize function
!(assertEqual (normalize 0.5) 0.5) 
!(assertEqual (normalize 2) 0.8807970779778823)  
!(assertEqual (normalize -2) 0.11920292202211757)

; Test normalizeWithSpecificRange
!(assertEqual (normalizeWithSpecificRange 0.5 0 10) 0.5)  
!(assertEqual (normalizeWithSpecificRange 2 0 10) 2)
!(assertEqual (normalizeWithSpecificRange -2 0 10) 1.1920292202211757)  
!(assertEqual (normalizeWithSpecificRange 5 10 20) 19.93307149075715)

; Test sumAtom
!(assertEqual (sumAtom (5 4 3)) 12)
!(assertEqual (sumAtom ()) 0)  
!(assertEqual (sumAtom (1)) 1)

; Test mean
!(assertEqual (mean (5 4 3)) 4)
!(assertEqual (mean ()) 0.0)  
!(assertEqual (mean (2)) 2)

; Test sumSquaredDifference
!(assertEqual (sumSquaredDifference (5 4 3) 4) 2)
!(assertEqual (sumSquaredDifference (2) 2) 0)  
!(assertEqual (sumSquaredDifference () 0) 0) 

; Test standardDeviation
!(assertEqual (standardDeviation (5 4 3)) 0.816496580927726)  
!(assertEqual (standardDeviation ()) ())  
!(assertEqual (standardDeviation (2)) 0.0)
!(assertEqual (standardDeviation (2 2 2)) 0.0)

; Test normalizeUsingMinMax
!(assertEqual (normalizeUsingMinMax 6.0 1 10) 0.5555555555555556)   
!(assertEqual (normalizeUsingMinMax 0.5 0 1) 0.5)   
!(assertEqual (normalizeUsingMinMax 5 5 5) 0)   

; Test normalizeList
!(assertEqualToResult (normalizeList (5 4 3)) (1.0 0.5 0.0))   
!(assertEqual (normalizeList ()) ())   
!(assertEqualToResult (normalizeList (2)) (0.0))   
!(assertEqualToResult (normalizeList (2 2 2)) (0.0 0.0 0.0)) 

; Test normalizeUsingZScore
!(assertEqual (normalizeUsingZScore 6.0 4 0.816496580927726) 2.449489742783178)   
!(assertEqual (normalizeUsingZScore 0.5 0 1) 0.5)   
!(assertEqual (normalizeUsingZScore 5 5 0) 0)   

; Test normalizeUsingZScoreDataset
!(assertEqual (normalizeUsingZScoreDataset 6.0 (5 4 3)) 2.449489742783178)  
!(assertEqual (normalizeUsingZScoreDataset 0.5 (5 4 3)) -4.286607049870562)   
!(assertEqual (normalizeUsingZScoreDataset 2 ()) ())   
!(assertEqual (normalizeUsingZScoreDataset 2 (2)) 0.0)   
!(assertEqual (normalizeUsingZScoreDataset 2 (2 2 2)) 0)