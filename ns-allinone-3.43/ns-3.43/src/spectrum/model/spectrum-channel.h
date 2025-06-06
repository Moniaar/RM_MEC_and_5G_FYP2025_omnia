/*
 * Copyright (c) 2009 CTTC
 *
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Author: Nicola Baldo <nbaldo@cttc.es>
 */

#ifndef SPECTRUM_CHANNEL_H
#define SPECTRUM_CHANNEL_H

#include "phased-array-spectrum-propagation-loss-model.h"
#include "spectrum-phy.h"
#include "spectrum-propagation-loss-model.h"
#include "spectrum-signal-parameters.h"
#include "spectrum-transmit-filter.h"

#include <ns3/channel.h>
#include <ns3/mobility-model.h>
#include <ns3/nstime.h>
#include <ns3/object.h>
#include <ns3/propagation-delay-model.h>
#include <ns3/propagation-loss-model.h>
#include <ns3/traced-callback.h>

namespace ns3
{

class PacketBurst;
class SpectrumValue;

/**
 * \ingroup spectrum
 *
 * Defines the interface for spectrum-aware channel implementations
 *
 */
class SpectrumChannel : public Channel
{
  public:
    /**
     * constructor
     *
     */
    SpectrumChannel();
    /**
     * destructor
     *
     */
    ~SpectrumChannel() override;

    // inherited from Object
    void DoDispose() override;

    /**
     * \brief Get the type ID.
     * \return the object TypeId
     */
    static TypeId GetTypeId();

    /**
     * Add the single-frequency propagation loss model to be used
     * \warning only models that do not depend on the TX power should be used.
     *
     * \param loss a pointer to the propagation loss model to be used.
     */
    void AddPropagationLossModel(Ptr<PropagationLossModel> loss);

    /**
     * Add the frequency-dependent propagation loss model to be used
     * \param loss a pointer to the propagation loss model to be used.
     */
    void AddSpectrumPropagationLossModel(Ptr<SpectrumPropagationLossModel> loss);

    /**
     * Add the frequency-dependent propagation loss model
     * that is compapatible with the phased antenna arrays at the TX and RX
     * \param loss a pointer to the propagation loss model to be used.
     */
    void AddPhasedArraySpectrumPropagationLossModel(
        Ptr<PhasedArraySpectrumPropagationLossModel> loss);

    /**
     * Set the propagation delay model to be used.  This method will abort
     * the simulation if there exists a previously set propagation delay model
     * (i.e., unlike propagation loss models, multiple of which can be chained
     * together, there is only one propagation delay model).
     * \param delay Ptr to the propagation delay model to be used.
     */
    void SetPropagationDelayModel(Ptr<PropagationDelayModel> delay);

    /**
     * Get the frequency-dependent propagation loss model.
     * \returns a pointer to the propagation loss model.
     */
    Ptr<SpectrumPropagationLossModel> GetSpectrumPropagationLossModel() const;

    /**
     * Get the frequency-dependent propagation loss model that is
     * compatible with the phased antenna arrays at TX and RX
     * \returns a pointer to the propagation loss model.
     */
    Ptr<PhasedArraySpectrumPropagationLossModel> GetPhasedArraySpectrumPropagationLossModel() const;

    /**
     * Get the propagation loss model.
     * \returns a pointer to the propagation loss model.
     */
    Ptr<PropagationLossModel> GetPropagationLossModel() const;

    /**
     * Get the propagation delay model that has been set on the channel.
     * \returns a pointer to the propagation delay model.
     */
    Ptr<PropagationDelayModel> GetPropagationDelayModel() const;

    /**
     * Add the transmit filter to be used to filter possible signal receptions
     * at the StartTx() time.  This method may be called multiple
     * times to chain multiple filters together; the last filter added will
     * be the first one used in the chain.
     *
     * \param filter an instance of a SpectrumTransmitFilter
     */
    void AddSpectrumTransmitFilter(Ptr<SpectrumTransmitFilter> filter);

    /**
     * Get the transmit filter, or first in a chain of transmit filters
     * if more than one is present.
     * \returns a pointer to the transmit filter.
     */
    Ptr<SpectrumTransmitFilter> GetSpectrumTransmitFilter() const;

    /**
     * Used by attached PHY instances to transmit signals on the channel
     *
     * \param params the parameters of the signals being transmitted
     */
    virtual void StartTx(Ptr<SpectrumSignalParameters> params) = 0;

    /**
     * This method calls AssignStreams() on any/all of the PropagationLossModel,
     * PropagationDelayModel, SpectrumPropagationLossModel,
     * PhasedArraySpectrumPropagationLossModel, and SpectrumTransmitFilter
     * objects that have been added to this channel.  If any of those
     * objects are chained together (e.g., multiple PropagationDelayModel
     * objects), the base class of that object is responsible for ensuring
     * that all models in the chain have AssignStreams recursively called.
     *
     * \param stream the stream index offset start
     * \return the number of stream indices assigned by this model
     */
    int64_t AssignStreams(int64_t stream);

    /**
     * \brief Remove a SpectrumPhy from a channel
     *
     * This method is used to detach a SpectrumPhy instance from a
     * SpectrumChannel instance, so that the SpectrumPhy does not receive
     * packets sent on that channel.
     *
     * This method is to be implemented by all classes inheriting from
     * SpectrumChannel.
     *
     * @param phy the SpectrumPhy instance to be removed from the channel as
     * a receiver.
     */
    virtual void RemoveRx(Ptr<SpectrumPhy> phy) = 0;

    /**
     * \brief Add a SpectrumPhy to a channel, so it can receive packets
     *
     * This method is used to attach a SpectrumPhy instance to a
     * SpectrumChannel instance, so that the SpectrumPhy can receive
     * packets sent on that channel. Note that a SpectrumPhy that only
     * transmits (without receiving ever) does not need to be added to
     * the channel.
     *
     * This method is to be implemented by all classes inheriting from
     * SpectrumChannel.
     *
     * \param phy the SpectrumPhy instance to be added to the channel as
     * a receiver.
     */
    virtual void AddRx(Ptr<SpectrumPhy> phy) = 0;

    /**
     * TracedCallback signature for path loss calculation events.
     *
     * \param [in] txPhy The TX SpectrumPhy instance.
     * \param [in] rxPhy The RX SpectrumPhy instance.
     * \param [in] lossDb The loss value, in dB.
     */
    typedef void (*LossTracedCallback)(Ptr<const SpectrumPhy> txPhy,
                                       Ptr<const SpectrumPhy> rxPhy,
                                       double lossDb);
    /**
     * TracedCallback signature for path loss calculation events.
     *
     * \param [in] txMobility The mobility model of the transmitter.
     * \param [in] rxMobility The mobility model of the receiver.
     * \param [in] txAntennaGain The transmitter antenna gain, in dB.
     * \param [in] rxAntennaGain The receiver antenna gain, in dB.
     * \param [in] propagationGain The propagation gain, in dB.
     * \param [in] pathloss The path loss value, in dB.
     */
    typedef void (*GainTracedCallback)(Ptr<const MobilityModel> txMobility,
                                       Ptr<const MobilityModel> rxMobility,
                                       double txAntennaGain,
                                       double rxAntennaGain,
                                       double propagationGain,
                                       double pathloss);
    /**
     * TracedCallback signature for Ptr<const SpectrumSignalParameters>.
     *
     * \param [in] params SpectrumSignalParameters instance.
     */
    typedef void (*SignalParametersTracedCallback)(Ptr<SpectrumSignalParameters> params);

  protected:
    /**
     * This provides a base class implementation that may be subclassed
     * if needed by subclasses that might need additional stream assignments.
     *
     * \param stream first stream index to use
     * \return the number of stream indices assigned by this model
     */
    virtual int64_t DoAssignStreams(int64_t stream);

    /**
     * The `PathLoss` trace source. Exporting the pointers to the Tx and Rx
     * SpectrumPhy and a pathloss value, in dB.
     */
    TracedCallback<Ptr<const SpectrumPhy>, Ptr<const SpectrumPhy>, double> m_pathLossTrace;

    /**
     * The `Gain` trace source. Fired whenever a new path loss value
     * is calculated. Exporting pointer to the mobility model of the transmitter and
     * the receiver, Tx antenna gain, Rx antenna gain, propagation gain and pathloss
     */
    TracedCallback<Ptr<const MobilityModel>,
                   Ptr<const MobilityModel>,
                   double,
                   double,
                   double,
                   double>
        m_gainTrace;

    /**
     * Traced callback for SpectrumSignalParameters in StartTx requests
     */
    TracedCallback<Ptr<SpectrumSignalParameters>> m_txSigParamsTrace;

    /**
     * Maximum loss [dB].
     *
     * Any device above this loss is considered out of range.
     */
    double m_maxLossDb;

    /**
     * Single-frequency propagation loss model to be used with this channel.
     */
    Ptr<PropagationLossModel> m_propagationLoss;

    /**
     * Propagation delay model to be used with this channel.
     */
    Ptr<PropagationDelayModel> m_propagationDelay;

    /**
     * Frequency-dependent propagation loss model to be used with this channel.
     */
    Ptr<SpectrumPropagationLossModel> m_spectrumPropagationLoss;

    /**
     * Frequency-dependent propagation loss model to be used with this channel.
     */
    Ptr<PhasedArraySpectrumPropagationLossModel> m_phasedArraySpectrumPropagationLoss;

    /**
     * Transmit filter to be used with this channel
     */
    Ptr<SpectrumTransmitFilter> m_filter{nullptr};
};

} // namespace ns3

#endif /* SPECTRUM_CHANNEL_H */
