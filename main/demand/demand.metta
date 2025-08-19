!(bind! &demandspace (new-space))
(: Demand Type)
(: SkipDemand Type)
; (: demand (-> Symbol Number Demand))
(: skip (-> Symbol SkipDemand))

; (: addDemand (-> hyperon::space::DynSpace Symbol Number Demand))
(= (addDemand $space $name $value)
    (let* (
            ($demand (demand $name $value))
            ; ($duplicateDemandList  (match $space (demand $name $v) (demand $name $v)))
            (() (add-atom $space $demand))
        )
     ()
    )
    )

!(addDemand &demandspace energy 0.2)

(= (getAllDemands $space)
    (collapse (match $space (demand $x $_) (demand $x $_)))
)
!(getAllDemands &self)
(= (skipDemand $space $demand ) 
    (let* (
        ($skip (collapse (match $space (skip $demand) (skip $demand))))
        ( $isNotFound (== $skip ()))
    )
    (if $isNotFound 
        (add-reduct $space (skip $demand)) 
        ()
    )
    )
)

!(skipDemand &self energy)
; !(add-reduct &demandspace (superpose (
;         (skip competence 0.5)
; )))

(= (getEnabledDemands $space)
    (let* (
       ( $skippedDemands (collapse (match $space (, (demand $x $_) (skip $x)) (demand $x $_)) ))
    
       ( $allDemands (getAllDemands $space))
    
       ( $enabledDemands (collapse (subtraction  (superpose $allDemands)  (superpose $skippedDemands))))
     
    ) 
        $enabledDemands
    )
)

(= (setDemandValue $space $demandName $newValue)
    (let* (
        ((demand $demandName $oldValue ) (match $space (demand $demandName $oldValue) (demand $demandName $oldValue)))
        (() (remove-atom $space (demand $demandName $oldValue)))
        (() (add-atom $space (demand $demandName $newValue)))
    )
    ()
    )

)

; Fixed DemandDecay function to properly take space parameter
(= (DemandDecay $space $demand $value)
    (let* (
        ($oldValue (collapse (match $space (demand $demand $oldValue) $oldValue)))
        ($newValue (- $oldValue $value))
    )
    (if (>= $newValue 0)
        (setDemandValue $space $demand $newValue)
        (let (($demandAtom (collapse (match $space (demand $demand $oldValue) (demand $demand $oldValue)))))
            (remove-atom $space $demandAtom)
        )
    )
    )
)

; New function to apply decay only to skipped demands
(= (DecaySkippedDemands $space $decayValue)
    (let* (
        ($skippedDemandNames (collapse (match $space (skip $x) $x)))
        ($skippedDemands (collapse (match $space (demand $x $value) (demand $x $value))))
    )
    (if (== (size-atom $skippedDemands) 0 )
        (There is no skipped demand) 
        (superpose (DemandDecay $space (superpose $skippedDemandNames) $decayValue))
    ))
)
!(DecaySkippedDemands &self 0.1)
; New function to apply decay to all demands (both enabled and skipped)
(= (DecayAllDemands $space $decayValue)
    (let* (
        ($allDemands (getAllDemands $space))
        ($demandNames (collapse (match $space (demand $x $_) $x)))
    )
    (if (== (size-atom $allDemands) 0 )
        (No demands in the space)
        (superpose (DemandDecay $space (superpose $demandNames) $decayValue))
    ))
)

; New function to apply different decay rates to enabled vs skipped demands
(= (DecayDemandsWithDifferentRates $space $enabledDecayRate $skippedDecayRate)
    (let* (
        ($enabledDemandNames (collapse (match $space (demand $x $_) 
            (if (== (collapse (match $space (skip $x) (skip $x))) ()) $x ()))))
        ($skippedDemandNames (collapse (match $space (skip $x) $x)))
        ($_ (superpose (DemandDecay $space (superpose $enabledDemandNames) $enabledDecayRate)))
        ($_ (superpose (DemandDecay $space (superpose $skippedDemandNames) $skippedDecayRate)))
    )
    (
        ()
    ))
)

!(skipDemand energy)
!(skipDemand affiliation)

!(setDemandValue energy 0.6)


!(getEnabledDemands)


!(get-atoms &demandspace)