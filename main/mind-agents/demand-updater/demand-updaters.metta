; Demand Updater - Based on OpenPsi ben paper
; util functions

; fuzzy new
; function used to compute certainty demand level
; ti is the latest time stamp of observing object i and
; ts is the current time stamp of the virtual world.
(= (fuzzy-new $ti $ts)
    (let $x (exp (* 0.002 (- $ts $ti)))
    (/ 2 (+ 1 $x))))

; fuzzy near
; function used to compute affiliation demand level
; di - denotes the distance between friend i and the avatar itself
; dmax - is a distance threshold to decrease the impact of friends far away.
(= (fuzzy-near $di $dmax)
    (/ 1 (+ 1 (* 0.00015 (- $di $dmax))))
)
; summation of fuzzy near
(= (sum-fuzzy-near $lst $d-max)
    (let $fnear (collapse (fuzzy-near (superpose $lst) $d-max))
    (fold + 0 $fnear))
)
; summation of fuzzy new
(= (sum-fuzzy-new $lst $ts)
    (let $fnew (collapse (fuzzy-new (superpose $lst) $ts))
    (fold + 0 $fnew))
)

; Certainty Demand Updater
;(: CertaintyDemandUpdater (-> Expression Number Number))
(= (CertaintyDemandUpdater $lst $ts)
    (let* (
            ($summation (sum-fuzzy-new $lst $ts))
            ($object_num (size-atom $lst))
            ($rand (random-float 0.0 1.0)) ; Assuming random-float is available
            ($x (exp (* -0.05 $object-num)))
        )
        (/ (+ $summation $rand) (+ 1 (* $x (+ 1 $object_num))))
    )
)


; Affiliation Demand Updater
(: AffiliationDemandUpdater (-> Expression Number Number))
(= (AffiliationDemandUpdater $lst $d-max)
    (let* (
            ($summation (sum-fuzzy-near $lst $d-max))
            ($friend-num (size-atom $lst))
            ($rand (random-float 0.0 1.0)) ; Assuming random-float is available
            ($x (exp (* -0.1 $friend-num)))
        )
        (/ (+ $summation $rand) (+ 1 (* $x (+ 1 $friend-num))))
    )
)

; Competence Demand Updater
(: CompetenceDemandUpdater (-> Number Number Number))
(= (CompetenceDemandUpdater $plan-done-number $plan-failed-number)
    (/ $plan-done-number (+ $plan-done-number (pow-math $plan-failed-number 1.5)))
)
