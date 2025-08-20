 
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

; Test normalize_withSpecificRange
!(assertEqual (normalize_withSpecificRange 0.5 0 10) 0.5)  
!(assertEqual (normalize_withSpecificRange 2 0 10) 2)
!(assertEqual (normalize_withSpecificRange -2 0 10) 1.1920292202211757)  
!(assertEqual (normalize_withSpecificRange 5 10 20) 19.93307149075715)

; Test sum-atom
!(assertEqual (sum-atom (5 4 3)) 12)
!(assertEqual (sum-atom ()) 0)  
!(assertEqual (sum-atom (1)) 1)

; Test mean
!(assertEqual (mean (5 4 3)) 4)
!(assertEqual (mean ()) 0.0)  
!(assertEqual (mean (2)) 2)

; Test sum_sq_diff
!(assertEqual (sum_sq_diff (5 4 3) 4) 2)
!(assertEqual (sum_sq_diff (2) 2) 0)  
!(assertEqual (sum_sq_diff () 0) 0) 

; Test std
!(assertEqual (std (5 4 3)) 0.816496580927726)  
!(assertEqual (std ()) ())  
!(assertEqual (std (2)) 0.0)
!(assertEqual (std (2 2 2)) 0.0)

; Test normalize_usingMinMax
!(assertEqual (normalize_usingMinMax 6.0 1 10) 0.5555555555555556)   
!(assertEqual (normalize_usingMinMax 0.5 0 1) 0.5)   
!(assertEqual (normalize_usingMinMax 5 5 5) 0)   

; Test normalize-list
!(assertEqualToResult (normalize-list (5 4 3)) (1.0 0.5 0.0))   
!(assertEqual (normalize-list ()) ())   
!(assertEqualToResult (normalize-list (2)) (0.0))   
!(assertEqualToResult (normalize-list (2 2 2)) (0.0 0.0 0.0)) 

; Test normalize_usingZScore
!(assertEqual (normalize_usingZScore 6.0 4 0.816496580927726) 2.449489742783178)   
!(assertEqual (normalize_usingZScore 0.5 0 1) 0.5)   
!(assertEqual (normalize_usingZScore 5 5 0) 0)   

; Test normalize_usingZScore_dataset
!(assertEqual (normalize_usingZScore_dataset 6.0 (5 4 3)) 2.449489742783178)  
!(assertEqual (normalize_usingZScore_dataset 0.5 (5 4 3)) -4.286607049870562)   
!(assertEqual (normalize_usingZScore_dataset 2 ()) ())   
!(assertEqual (normalize_usingZScore_dataset 2 (2)) 0.0)   
!(assertEqual (normalize_usingZScore_dataset 2 (2 2 2)) 0)