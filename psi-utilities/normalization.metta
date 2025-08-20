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
    (let* (($min (min-atom $list))
          ($max (max-atom $list))
          ($normalized
            (if-equal $min $max
                (map-atom $list $x 0.0)  
                (map-atom $list $x (/ (- $x $min) (- $max $min))))))
        (superpose $normalized)
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
      0  
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