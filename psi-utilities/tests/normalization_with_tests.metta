;using sigmoid function
;doesn't require dataset statistics 

;declare e, euler's number 
(= (e) 2.718281828459045)
;sigmoid function
(= (sigmoid $x)
   (/ 1.0 (+ 1.0 (pow-math (e) (* -1.0 $x))))
)
;normalization function
(= (normalize $x)
   (if (and (>= $x 0.0) (<= $x 1.0))
       $x
       (sigmoid $x)
    )
)
; normalize to range [a, b]
(= (normalize_withSpecificRange $x $a $b)
   (if (and (>= $x $a) (<= $x $b)) $x (+ $a (* (- $b $a) (sigmoid $x))))
)
;normalization using minmax
;normalize single value given min and max 
(= (normalize_usingMinMax $x $min $max)
  (if (== $min $max)
    0
    (if (and (>= $x 0) (<= $x 1))
      $x
      (/ (- $x $min) (- $max $min))
    )
  )
)
;normalize list of values using min-max
;main
;base-case
(= (normalize-list ()) ())
(= (normalize-list $list)
  (if-equal (size-atom $list) [0]
    ()  ; Empty list returns empty
    (let $min (min-atom $list)
      (let $max (max-atom $list)
        (let $normalized
          (if-equal $min $max
            (map-atom $list $x 0.0)  ; If min == max, return list of zeros
            (map-atom $list $x (/ (- $x $min) (- $max $min))))
          (superpose $normalized)
        )
      )
    )
  )
)   

; define max-atom 
(= (max-atom $list)
  (if-equal (size-atom $list) [0]
    [(Error (max-atom $list) EmptyExpression)]
    (foldl-atom $list (index-atom $list 0) $acc $x (if (> $x $acc) $x $acc))
  )
)

;normalize using z-score
;given value,mean,standard deviation 
(= (normalize_usingZScore $x $mean $std)
  (if (== $std 0)
    0
    (if (and (>= $x 0) (<= $x 1))
      $x
      (let $z (/ (- $x $mean) $std)
        $z
      )
    )
  )
)
;normalize using z-score given value and dataset
; Compute sum of list using foldl-atom
(= (sum-atom $list)
  (foldl-atom $list 0 $acc $x (+ $acc $x)))

;mean
(= (mean $list)
  (let $len (size-atom $list)
    (if (== $len 0)
      ()   
      (/ (sum-atom $list) $len)
    )
  )
)

;sum of squared differences  
(= (sum_sq_diff $list $mean)
  (sum-atom (map-atom $list $x (* (- $x $mean) (- $x $mean))))
)

;standard deviation
(= (std $list)
  (let $m (mean $list)
    (if (== $m ())
      ()   
      (let $len (size-atom $list)
        (if (== $len 0)
          ()   
          (sqrt-math (/ (sum_sq_diff $list $m) $len))
        )
      )
    )
  )
)
;main
(= (normalize_usingZScore_dataset $x $data)
  (let $m (mean $data)
    (if (== $m ())
      ()   
      (let $s (std $data)
        (if (== $s ())
          ()   
          (if (== $s 0)
            0  ; Zero std returns 0
            (/ (- $x $m) $s))
        )
      )
    )
  )
)
;;;;;;;;;;;;;;;;;;;;;;;;;; TEST ;;;;;;;;;;;;;;;;;;;;;;;;;
; Test  normalization functions

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
!(assertEqual (mean ()) ())  
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